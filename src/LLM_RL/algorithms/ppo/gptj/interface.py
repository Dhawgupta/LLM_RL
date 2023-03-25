from __future__ import annotations
import jax
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from functools import partial
from typing import List, Optional, Union, Tuple, Callable, NamedTuple, Dict, Any
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
import flax.linen as nn
from LLM_RL.algorithms.ppo.base_interface import PPOTrain, ppo_loss_fn, PPOInference, PPOForwardOutput
from jax.sharding import NamedSharding

class GPTJPPOTrain(PPOTrain):
    @classmethod
    def load_train(
        cls, 
        policy_train_state: TrainState, 
        value_head_train_state: TrainState, 
        policy_model: FlaxPreTrainedModel, 
        value_head_model: nn.Module, 

        tokenizer: PreTrainedTokenizerBase, 
        mesh: jax.sharding.Mesh, # mesh should have shape ('dp', 'mp')
        loss_fn: Callable=ppo_loss_fn, 
    ) -> GPTJPPOTrain:
        policy_train_state_partition_spec = match_partition_rules(policy_model.config.get_partition_rules(), policy_train_state)
        value_head_train_state_partition_spec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_train_state)

        @partial(
            jax.jit, 
            donate_argnums=(0, 1), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_train_state_partition_spec), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            policy_train_state: TrainState, 
            value_head_train_state: TrainState, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            old_logprobs: jax.Array, 
            old_values: jax.Array, 
            old_advantages: jax.Array, 
            old_returns: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, TrainState, jax.Array, PyTree]:
            # define loss function
            def grad_loss(policy_params: PyTree, value_head_params: PyTree, prng_key: jax.random.PRNGKeyArray):
                
                new_key = None
                if new_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                model_output = policy_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=policy_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )

                new_key = None
                if new_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                values = value_head_model.apply(
                    {'params': value_head_params}, 
                    model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key}, 
                )[:, :-1]

                logits = model_output.logits
                logits = logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(float('-inf'))
                logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])

                loss, info = loss_fn(
                    attention_mask, 
                    logprobs, 
                    values, 
                    should_take_action, 
                    old_logprobs, 
                    old_values, 
                    old_advantages, 
                    old_returns, 
                )
                return loss, info
            train_state = train_state
            # take loss
            (loss, info), (policy_grads, value_head_grads) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0, 1))(
                policy_train_state.params, 
                value_head_train_state.params, 
                prng_key, 
            )
            # assert shard gradients
            policy_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                policy_grads, 
                policy_train_state_partition_spec.params, 
            )
            value_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                value_head_grads, 
                value_head_train_state_partition_spec.params, 
            )
            # update params and optim state
            policy_train_state = policy_train_state.apply_gradients(grads=policy_grads)
            value_head_train_state = value_head_train_state.apply_gradients(grads=value_head_grads)

            return policy_train_state, value_head_train_state, loss, info
        
        return cls(
            policy_train_state=policy_train_state, 
            value_head_train_state=value_head_train_state, 
            policy_model=policy_model, 
            value_head_model=value_head_model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )

class GPTJPPOInference(PPOInference):
    @classmethod
    def load_inference(
        cls, 
        initial_policy_params: PyTree, 
        policy_params: PyTree, 
        value_head_params: PyTree, 
        initial_policy_model: FlaxPreTrainedModel, 
        policy_model: FlaxPreTrainedModel, 
        value_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        mesh: jax.sharding.Mesh, # mesh should have shape ('dp', 'mp')
        loss_fn: Optional[Callable]=ppo_loss_fn, 
    ) -> GPTJPPOInference:
        initial_policy_params_partition_spec = match_partition_rules(initial_policy_model.config.get_partition_rules(), initial_policy_params)
        policy_params_partition_spec = match_partition_rules(initial_policy_model.config.get_partition_rules(), policy_params)
        value_head_params_partition_spec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_params)

        @partial(
            jax.jit, 
            static_argnames=('initial_policy_output_attentions', 'initial_policy_output_hidden_states', 'policy_output_attentions', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), initial_policy_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_params_partition_spec), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                PPOForwardOutput(
                    initial_policy_raw_output=FlaxCausalLMOutput(
                        logits=NamedSharding(mesh, PS("dp", None, None)), 
                        hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                        attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                    ), 
                    policy_raw_output=FlaxCausalLMOutput(
                        logits=NamedSharding(mesh, PS("dp", None, None)), 
                        hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                        attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                    ), 
                    values=NamedSharding(mesh, PS("dp", None)), 
                ), 
            ), 
        )
        def _forward(
            initial_policy_params: PyTree, 
            policy_params: PyTree, 
            value_head_params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            initial_policy_output_attentions: Optional[bool]=None, 
            initial_policy_output_hidden_states: Optional[bool]=None, 
            policy_output_attentions: Optional[bool]=None, # no policy_output_hidden_states option because this is required
            train: bool=False, 
        ) -> PPOForwardOutput:
            
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            initial_model_output = initial_policy_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=initial_policy_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=initial_policy_output_hidden_states, 
                output_attentions=initial_policy_output_attentions, 
            )
            model_output = policy_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=policy_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
                output_attentions=policy_output_attentions, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            values = value_head_model.apply(
                {'params': value_head_params}, 
                model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key}, 
            )

            return PPOForwardOutput(
                initial_policy_raw_output=initial_model_output, 
                policy_raw_output=model_output, 
                values=values, 
            )
    
        @partial(
            jax.jit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_params_partition_spec), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS("dp", None)), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _eval_loss(
            policy_params: PyTree, 
            value_head_params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            old_logprobs: jax.Array, 
            old_values: jax.Array, 
            old_advantages: jax.Array, 
            old_returns: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            model_output = policy_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=policy_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            values = value_head_model.apply(
                {'params': value_head_params}, 
                model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key}, 
            )[:, :-1]

            logits = model_output.logits
            logits = logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(float('-inf'))
            logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])

            loss, info = loss_fn(
                attention_mask, 
                logprobs, 
                values, 
                should_take_action, 
                old_logprobs, 
                old_values, 
                old_advantages, 
                old_returns, 
            )
            return loss, info
    
        return cls(
            initial_policy_params=initial_policy_params, 
            policy_params=policy_params, 
            value_head_params=value_head_params, 
            initial_policy_model=initial_policy_model, 
            policy_model=policy_model, 
            value_head_model=value_head_model, 
            tokenizer=tokenizer, 
            _forward=_forward, 
            _eval_loss=_eval_loss, 
        )
