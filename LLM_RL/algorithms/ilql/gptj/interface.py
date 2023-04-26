from typing import Optional, Callable, Tuple, Union, List
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.ilql.base_interface import ILQLTrain, ILQLInferenceFull, ILQLInferenceSimple
from flax.training.train_state import TrainState
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, BlockingStrategy, block_sequences, Padding, Truncation
from functools import partial
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from LLM_RL.algorithms.ilql.gptj.generation import CheapGPTJILQLGeneration, FullGPTJILQLGeneration
from LLM_RL.algorithms.ilql.base_interface import ILQLSimpleForwardOutput, ILQLFullForwardOutput
from JaxSeq.stream_tokens import StreamingGenerationConfig
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from LLM_RL.algorithms.ilql.base_interface import ILQLPolicy
from transformers.generation import GenerationConfig
from LLM_RL.environment import TextHistory, text_history_to_str, Text
from examples_jaxseq.misc.commandline_server_client import strip_prompt_from_completion

class GPTJILQLTrain(ILQLTrain):
    @classmethod
    def load_train(
        cls, 
        base_train_state: TrainState, 
        target_base_params: Optional[PyTree], 
        q1_head_train_state: TrainState, 
        q2_head_train_state: TrainState, 
        v_head_train_state: TrainState, 
        q1_target_head_params: PyTree, 
        q2_target_head_params: PyTree, 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        v_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        detach_q1: bool, 
        detach_q2: bool, 
        detach_v: bool, 
        polyak_alpha: float, 
        hard_update_every: Optional[int], 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        assert mesh == v_head_model.config.mesh
        base_train_state_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_train_state)
        target_base_params_partition_spec = PS() if target_base_params is None else match_partition_rules(base_model.config.get_partition_rules(), target_base_params)
        q1_head_train_state_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_head_train_state)
        q2_head_train_state_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_head_train_state)
        v_head_train_state_partition_spec = match_partition_rules(v_head_model.config.get_partition_rules(), v_head_train_state)
        q1_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_target_head_params)
        q2_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_target_head_params)

        @partial(
            pjit, 
            donate_argnums=(0, 1, 2, 3, 4, 5, 6), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            base_train_state: TrainState, 
            target_base_params: Optional[PyTree], 
            q1_head_train_state: TrainState, 
            q2_head_train_state: TrainState, 
            v_head_train_state: TrainState, 
            q1_target_head_params: PyTree, 
            q2_target_head_params: PyTree, 

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS('dp', None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS('dp', None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS('dp', None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS('dp', None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS('dp', None))
            dones = with_named_sharding_constraint(dones, mesh, PS('dp'))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS('dp', None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS('dp', None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS('dp', None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS('dp'))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None

            # define loss function

            def grad_loss(base_params: PyTree, q1_head_params: PyTree, q2_head_params: PyTree, v_head_params: PyTree, prng_key: jax.random.PRNGKeyArray):
                
                # get base hidden states

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                base_model_output = base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )

                if target_base_params is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    target_base_model_output = base_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids, 
                        params=target_base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=True, 
                    )
                else:
                    target_base_model_output = base_model_output
                
                if next_token_ids is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    next_token_base_model_output = base_model(
                        input_ids=next_token_ids, 
                        attention_mask=next_tokens_attention_mask, 
                        position_ids=next_tokens_position_ids, 
                        params=base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=True, 
                    )
                
                # get values

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q1_head_output = q_head_model.apply(
                    {'params': q1_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q2_head_output = q_head_model.apply(
                    {'params': q2_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                v_head_output = v_head_model.apply(
                    {'params': v_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_q1_head_output = q_head_model.apply(
                    {'params': q1_target_head_params}, 
                    target_base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_q2_head_output = q_head_model.apply(
                    {'params': q2_target_head_params}, 
                    target_base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                # stop gradients
                if detach_q1:
                    q1_head_output = jax.lax.stop_gradient(q1_head_output)
                if detach_q2:
                    q2_head_output = jax.lax.stop_gradient(q2_head_output)
                if detach_v:
                    v_head_output = jax.lax.stop_gradient(v_head_output)
                target_q1_head_output = jax.lax.stop_gradient(target_q1_head_output)
                target_q2_head_output = jax.lax.stop_gradient(target_q2_head_output)

                q1 = jnp.take_along_axis(q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                q2 = jnp.take_along_axis(q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                v = v_head_output[:, :-1].squeeze(2)
                v_full = v_head_output.squeeze(2)
                target_q1 = jnp.take_along_axis(target_q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                target_q2 = jnp.take_along_axis(target_q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)

                q1_logits = q1_head_output[:, :-1, :]
                q2_logits = q2_head_output[:, :-1, :]

                # get next token values

                if next_token_ids is not None:
                    # just run vf on last token to save some flops
                    last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_next_token_h = next_token_base_model_output.hidden_states[-1][jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]

                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    next_token_v_head_output = v_head_model.apply(
                        {'params': v_head_params}, 
                        final_next_token_h, 
                        train=train, 
                        rngs={'dropout': new_key} if prng_key is not None else None, 
                    ).sqeueze(1)
                    v_final = next_token_v_head_output * (1 - next_dones.astype(jnp.float32))
                else:
                    last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                    last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                    v_final = v_full[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                    v_final = v_final * (1 - dones)
                v_final = jax.lax.stop_gradient(v_final)

                loss, info = loss_fn(
                    q1, 
                    q2, 
                    v, 
                    v_final, 
                    target_q1, 
                    target_q2, 
                    q1_logits, 
                    q2_logits, 
                    input_ids[:, 1:], 
                    attention_mask[:, 1:], 
                    should_take_action, 
                    rewards, 
                )
                return loss, info

            # take loss
            (loss, info), (base_grads, q1_head_grads, q2_head_grads, v_head_grads) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0, 1, 2, 3))(
                base_train_state.params, 
                q1_head_train_state.params, 
                q2_head_train_state.params, 
                v_head_train_state.params, 
                prng_key, 
            )
            # assert shard gradients
            base_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                base_grads, 
                base_train_state_partition_spec.params, 
            )
            q1_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                q1_head_grads, 
                q1_head_train_state_partition_spec.params, 
            )
            q2_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                q2_head_grads, 
                q2_head_train_state_partition_spec.params, 
            )
            v_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                v_head_grads, 
                v_head_train_state_partition_spec.params, 
            )
            # update params and optim state
            base_train_state = base_train_state.apply_gradients(grads=base_grads)
            q1_head_train_state = q1_head_train_state.apply_gradients(grads=q1_head_grads)
            q2_head_train_state = q2_head_train_state.apply_gradients(grads=q2_head_grads)
            v_head_train_state = v_head_train_state.apply_gradients(grads=v_head_grads)

            # handle target network updates
            def update_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                base_params = optax.incremental_update(params, base_params, polyak_alpha)
                if hard_update_every is not None:
                    base_params = optax.periodic_update(params, base_params, steps, hard_update_every)
                return base_params
            
            def mid_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                return base_params

            def update_cond(opt_state: PyTree) -> bool:
                if hasattr(opt_state, 'mini_step'):
                    return opt_state.mini_step == 0
                return True
            
            if target_base_params is not None:
                target_base_params = jax.lax.cond(
                    update_cond(base_train_state.opt_state), 
                    update_targets, 
                    mid_targets, 
                    base_train_state.params, 
                    target_base_params, 
                    base_train_state.step, 
                )
            q1_target_head_params = jax.lax.cond(
                update_cond(q1_head_train_state.opt_state), 
                update_targets, 
                mid_targets, 
                q1_head_train_state.params, 
                q1_target_head_params, 
                q1_head_train_state.step, 
            )
            q2_target_head_params = jax.lax.cond(
                update_cond(q2_head_train_state.opt_state), 
                update_targets, 
                mid_targets, 
                q2_head_train_state.params, 
                q2_target_head_params, 
                q2_head_train_state.step, 
            )

            return base_train_state, target_base_params, q1_head_train_state, q2_head_train_state, v_head_train_state, q1_target_head_params, q2_target_head_params, loss, info

        return cls(
            base_train_state=base_train_state, 
            target_base_params=target_base_params, 
            q1_head_train_state=q1_head_train_state, 
            q2_head_train_state=q2_head_train_state, 
            v_head_train_state=v_head_train_state, 
            q1_target_head_params=q1_target_head_params, 
            q2_target_head_params=q2_target_head_params, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            v_head_model=v_head_model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )


class GPTJILQLInferenceSimple(ILQLInferenceSimple):
    @classmethod
    def load_inference(
        cls, 
        pi_beta_params: Optional[PyTree], 
        base_params: PyTree, 
        q1_head_params: PyTree, 
        q2_head_params: PyTree, 
        v_head_params: PyTree, 
        pi_beta_model: Optional[FlaxPreTrainedModel], 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        v_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        beta: float=0.0, 
        dp_shard_logits: bool=True, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        assert mesh == v_head_model.config.mesh
        assert (pi_beta_model is None and pi_beta_params is None) or (pi_beta_model is not None and pi_beta_params is not None)
        
        pi_beta_params_partition_spec = None
        if pi_beta_params is not None:
            pi_beta_params_partition_spec = match_partition_rules(pi_beta_model.config.get_partition_rules(), pi_beta_params)
        base_params_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_params)
        q1_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_head_params)
        q2_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_head_params)
        v_head_params_partition_spec = match_partition_rules(v_head_model.config.get_partition_rules(), v_head_params)

        generator = None
        if pi_beta_model is not None:
            generator = CheapGPTJILQLGeneration(
                base_model_config=base_model.config, 
                pi_beta=pi_beta_model, 
                value_base=base_model, 
                q_head=q_head_model, 
                v_head=v_head_model, 
                beta=beta, 
            )

        if pi_beta_params is not None:
            @partial(
                pjit, 
                static_argnames=('generation_config', 'trace'), 
                in_shardings=(
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                ), 
                out_shardings=NamedSharding(mesh, PS()), 
            )
            def _generate(
                pi_beta_params: PyTree, 
                base_params: PyTree, 
                q1_head_params: PyTree, 
                q2_head_params: PyTree, 
                v_head_params: PyTree, 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                # data parallel shard inputs
                input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
                attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
                position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))
                # NOTE: position_ids ignored by transformers

                # generate from model
                output = generator.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    params=(pi_beta_params, base_params, q1_head_params, q2_head_params, v_head_params), 
                    prng_key=prng_key, 
                    generation_config=StreamingGenerationConfig.from_dict(generation_config) if generation_config is not None else None, 
                    trace=trace, 
                )
                
                return output
        else:
            def _generate(
                pi_beta_params: PyTree, 
                base_params: PyTree, 
                q1_head_params: PyTree, 
                q2_head_params: PyTree, 
                v_head_params: PyTree, 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                raise NotImplementedError
        
        @partial(
            pjit, 
            static_argnames=('output_attentions', 'output_hidden_states', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=ILQLSimpleForwardOutput(
                base_raw_output=FlaxCausalLMOutput(
                    logits=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                ), 
                q1=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                q2=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                v=NamedSharding(mesh, PS()), 
            ), 
        )
        def _forward(
            base_params: PyTree, 
            q1_head_params: PyTree, 
            q2_head_params: PyTree, 
            v_head_params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            train: bool=False, 
        ) -> ILQLSimpleForwardOutput:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))

            # get logits
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=True, 
                dropout_rng=new_key, 
            )
            # trunc padded logits
            base_output = base_output.replace(logits=base_output.logits.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf')))

            # get q1
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1 = q_head_model.apply(
                {'params': q1_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q1 = q1.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get q2
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q2 = q_head_model.apply(
                {'params': q2_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q2 = q2.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get v
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            v = v_head_model.apply(
                {'params': v_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            ).squeeze(2)

            # assert sharding on outputs
            if dp_shard_logits:
                base_output = base_output.replace(logits=with_named_sharding_constraint(base_output.logits, mesh, PS("dp", None, None)))
                q1 = with_named_sharding_constraint(q1, mesh, PS("dp", None, None))
                q2 = with_named_sharding_constraint(q2, mesh, PS("dp", None, None))
            return ILQLSimpleForwardOutput(
                base_raw_output=base_output, 
                q1=q1, 
                q2=q2, 
                v=v, 
            )

        return cls(
            pi_beta_params=pi_beta_params, 
            base_params=base_params, 
            q1_head_params=q1_head_params, 
            q2_head_params=q2_head_params, 
            v_head_params=v_head_params, 
            pi_beta_model=pi_beta_model, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            v_head_model=v_head_model, 
            tokenizer=tokenizer, 
            _generate=_generate, 
            _forward=_forward,
        )

class GPTJInferenceFull(ILQLInferenceFull):
    @classmethod
    def load_inference(
        cls, 
        pi_beta_params: Optional[PyTree], 
        base_params: PyTree, 
        target_base_params: PyTree, 
        q1_head_params: PyTree, 
        q2_head_params: PyTree, 
        v_head_params: PyTree, 
        q1_target_head_params: PyTree, 
        q2_target_head_params: PyTree, 
        pi_beta_model: FlaxPreTrainedModel, 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        v_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        beta: float=0.0, 
        dp_shard_logits: bool=True, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        assert mesh == v_head_model.config.mesh
        assert (pi_beta_model is None and pi_beta_params is None) or (pi_beta_model is not None and pi_beta_params is not None)

        pi_beta_params_partition_spec = None
        if pi_beta_params is not None:
            pi_beta_params_partition_spec = match_partition_rules(pi_beta_model.config.get_partition_rules(), pi_beta_params)
        base_params_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_params)
        target_base_params_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), target_base_params)
        q1_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_head_params)
        q2_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_head_params)
        v_head_params_partition_spec = match_partition_rules(v_head_model.config.get_partition_rules(), v_head_params)
        q1_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_target_head_params)
        q2_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_target_head_params)

        generator = None
        if pi_beta_model is not None:
            generator = FullGPTJILQLGeneration(
                base_model_config=base_model.config, 
                pi_beta=pi_beta_model, 
                value_base=base_model, 
                q_head=q_head_model, 
                v_head=v_head_model, 
                beta=beta, 
            )

        if pi_beta_params is not None:
            @partial(
                pjit, 
                static_argnames=('generation_config', 'trace'), 
                in_shardings=(
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                ), 
                out_shardings=NamedSharding(mesh, PS()), 
            )
            def _generate(
                pi_beta_params: PyTree, 
                base_params: PyTree, 
                target_base_params: PyTree, 
                v_head_params: PyTree, 
                q1_target_head_params: PyTree, 
                q2_target_head_params: PyTree, 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                # data parallel shard inputs
                input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
                attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
                position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))
                # NOTE: position_ids ignored by transformers

                # generate from model
                output = generator.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    params=(pi_beta_params, base_params, v_head_params, target_base_params, q1_target_head_params, q2_target_head_params), 
                    prng_key=prng_key, 
                    generation_config=StreamingGenerationConfig.from_dict(generation_config) if generation_config is not None else None, 
                    trace=trace, 
                )
                
                return output
        else:
            def _generate(
                pi_beta_params: PyTree, 
                base_params: PyTree, 
                target_base_params: PyTree, 
                v_head_params: PyTree, 
                q1_target_head_params: PyTree, 
                q2_target_head_params: PyTree, 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                raise NotImplementedError
        
        @partial(
            pjit, 
            static_argnames=('output_attentions', 'output_hidden_states', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=ILQLFullForwardOutput(
                base_raw_output=FlaxCausalLMOutput(
                    logits=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                ), 
                target_base_raw_output=FlaxCausalLMOutput(
                    logits=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                ), 
                q1=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                q2=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                v=NamedSharding(mesh, PS()), 
                q1_target=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                q2_target=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
            ), 
        )
        def _forward(
            base_params: PyTree, 
            target_base_params: PyTree, 
            q1_head_params: PyTree, 
            q2_head_params: PyTree, 
            v_head_params: PyTree, 
            q1_target_head_params: PyTree, 
            q2_target_head_params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            train: bool=False, 
        ) -> ILQLSimpleForwardOutput:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))

            # get logits
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=True, 
                dropout_rng=new_key, 
            )
            # trunc padded logits
            base_output = base_output.replace(logits=base_output.logits.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf')))

            target_base_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=target_base_params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=True, 
                dropout_rng=new_key, 
            )
            # trunc padded logits
            target_base_output = target_base_output.replace(logits=target_base_output.logits.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf')))

            # get q1
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1 = q_head_model.apply(
                {'params': q1_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q1 = q1.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get q2
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q2 = q_head_model.apply(
                {'params': q2_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q2 = q2.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get v
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            v = v_head_model.apply(
                {'params': v_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            ).squeeze(2)

            # get q1_target
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1_target = q_head_model.apply(
                {'params': q1_target_head_params}, 
                target_base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q1_target = q1_target.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get q2_target
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q2_target = q_head_model.apply(
                {'params': q2_target_head_params}, 
                target_base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q2_target = q2_target.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # assert sharding on outputs
            if dp_shard_logits:
                base_output = base_output.replace(logits=with_named_sharding_constraint(base_output.logits, mesh, PS("dp", None, None)))
                q1 = with_named_sharding_constraint(q1, mesh, PS("dp", None, None))
                q2 = with_named_sharding_constraint(q2, mesh, PS("dp", None, None))
                q1_target = with_named_sharding_constraint(q1_target, mesh, PS("dp", None, None))
                q2_target = with_named_sharding_constraint(q2_target, mesh, PS("dp", None, None))
            return ILQLFullForwardOutput(
                base_raw_output=base_output, 
                target_base_raw_output=target_base_output, 
                q1=q1, 
                q2=q2, 
                v=v, 
                q1_target=q1_target, 
                q2_target=q2_target, 
            )
        
        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _eval_loss(
            base_params: PyTree, 
            target_base_params: Optional[PyTree], 
            q1_head_params: PyTree, 
            q2_head_params: PyTree, 
            v_head_params: PyTree, 
            q1_target_head_params: PyTree, 
            q2_target_head_params: PyTree, 

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS('dp', None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS('dp', None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS('dp', None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS('dp', None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS('dp', None))
            dones = with_named_sharding_constraint(dones, mesh, PS('dp'))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS('dp', None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS('dp', None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS('dp', None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS('dp'))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None
                
            # get base hidden states

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_model_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
            )

            if target_base_params is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_base_model_output = base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=target_base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            else:
                target_base_model_output = base_model_output
            
            if next_token_ids is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                next_token_base_model_output = base_model(
                    input_ids=next_token_ids, 
                    attention_mask=next_tokens_attention_mask, 
                    position_ids=next_tokens_position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            
            # get values

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1_head_output = q_head_model.apply(
                {'params': q1_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q2_head_output = q_head_model.apply(
                {'params': q2_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            v_head_output = v_head_model.apply(
                {'params': v_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            target_q1_head_output = q_head_model.apply(
                {'params': q1_target_head_params}, 
                target_base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            target_q2_head_output = q_head_model.apply(
                {'params': q2_target_head_params}, 
                target_base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            # process outputs

            q1 = jnp.take_along_axis(q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            q2 = jnp.take_along_axis(q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            v = v_head_output[:, :-1].squeeze(2)
            v_full = v_head_output.squeeze(2)
            target_q1 = jnp.take_along_axis(target_q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            target_q2 = jnp.take_along_axis(target_q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)

            q1_logits = q1_head_output[:, :-1, :]
            q2_logits = q2_head_output[:, :-1, :]

            # get next token values

            if next_token_ids is not None:
                # just run vf on last token to save some flops
                last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_next_token_h = next_token_base_model_output.hidden_states[-1][jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                next_token_v_head_output = v_head_model.apply(
                    {'params': v_head_params}, 
                    final_next_token_h, 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                ).sqeueze(1)
                v_final = next_token_v_head_output * (1 - next_dones.astype(jnp.float32))
            else:
                last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                final_v = v_full[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                final_v = final_v * (1 - dones)

            loss, info = loss_fn(
                q1, 
                q2, 
                v, 
                v_final, 
                target_q1, 
                target_q2, 
                q1_logits, 
                q2_logits, 
                input_ids[:, 1:], 
                attention_mask[:, 1:], 
                should_take_action, 
                rewards, 
            )
            
            return loss, info

        return cls(
            pi_beta_params=pi_beta_params, 
            base_params=base_params, 
            target_base_params=target_base_params, 
            q1_head_params=q1_head_params, 
            q2_head_params=q2_head_params, 
            v_head_params=v_head_params, 
            q1_target_head_params=q1_target_head_params, 
            q2_target_head_params=q2_target_head_params, 
            pi_beta_model=pi_beta_model, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            v_head_model=v_head_model, 
            tokenizer=tokenizer, 
            _generate=_generate, 
            _forward=_forward, 
            _eval_loss=_eval_loss, 
        )

class GPTJPolicy(ILQLPolicy):
    def __init__(
        self, 
        inference: Union[ILQLInferenceSimple, ILQLInferenceFull], 
        prng_key: Optional[jax.random.KeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        in_str_process: Optional[Callable[[str], str]]=None, 
        out_str_process: Optional[Callable[[str], str]]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ):
        self.inference = inference
        self.prng_key = prng_key
        self.generation_config = generation_config
        self.blocking_strategy = blocking_strategy
        self.in_str_process = in_str_process
        self.out_str_process = out_str_process
        self.input_token_process = input_token_process
        self.target_token_process = target_token_process
        if self.in_str_process is None:
            self.in_str_process = lambda x: x
        if self.out_str_process is None:
            self.out_str_process = lambda x: x
        self.trace = trace
    
    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[TextHistory]]:
        if done is None:
            done = [False]*len(text_history)
        # force eos_token for done sequences
        eos_token = self.inference.tokenizer.eos_token
        if self.generation_config is not None and self.generation_config.eos_token_id is not None:
            eos_token = self.inference.tokenizer.decode(self.generation_config.eos_token_id)
        if eos_token is None:
            eos_token = self.inference.tokenizer.pad_token
        if eos_token is None:
            eos_token = ''
        
        raw_input_strs = [
            eos_token if d else self.in_str_process(text_history_to_str(item)) \
                for item, d in zip(text_history, done)
        ]

        new_key = None
        if self.prng_key is not None:
            self.prng_key, new_key = jax.random.split(self.prng_key)
        model_outputs = self.inference.generate_from_str(
            input_strs=raw_input_strs, 
            prng_key=new_key, 
            blocking_strategy=self.blocking_strategy, 
            generation_config=self.generation_config, 
            input_token_process=self.input_token_process, 
            target_token_process=self.target_token_process, 
            trace=self.trace, 
        )

        raw_output_strs = model_outputs.output_strs
        output_strs = [
            "" if d else self.out_str_process(strip_prompt_from_completion(raw_input_str, raw_output_str)) \
                for raw_input_str, raw_output_str, d in zip(raw_input_strs, raw_output_strs, done)
        ]

        return [
            None if d else text_history_item+(Text(output_str, True),) \
                for text_history_item, output_str, d in zip(text_history, output_strs, done)
        ]
    
    def set_params(self, policy_params: PyTree) -> None:
        if isinstance(self.inference, ILQLInferenceSimple):
            pi_beta_params, base_params, q1_head_params, \
                q2_head_params, v_head_params = policy_params
            self.inference = self.inference.replace(
                pi_beta_params=pi_beta_params, 
                base_params=base_params, 
                q1_head_params=q1_head_params, 
                q2_head_params=q2_head_params, 
                v_head_params=v_head_params, 
            )
        elif isinstance(self.inference, ILQLInferenceFull):
            pi_beta_params, base_params, target_base_params, q1_head_params, \
                q2_head_params, v_head_params, q1_target_head_params, q2_target_head_params = policy_params
            self.inference = self.inference.replace(
                pi_beta_params=pi_beta_params, 
                base_params=base_params, 
                target_base_params=target_base_params, 
                q1_head_params=q1_head_params, 
                q2_head_params=q2_head_params, 
                v_head_params=v_head_params, 
                q1_target_head_params=q1_target_head_params, 
                q2_target_head_params=q2_target_head_params, 
            )
        else:
            raise NotImplementedError
