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
from LLM_RL.environment import TextPolicy, TextHistory, text_history_to_str, Text
from JaxSeq.models.base_interface import Inference
from JaxSeq.utils import BlockingStrategy, Padding, Truncation
from transformers.generation import GenerationConfig
from JaxSeq.models.gptj.interface import GPTJInference
import jax.numpy as jnp
from LLM_RL.algorithms.ppo.base_interface import PPOPolicy
from jax.experimental.pjit import pjit

class GPTJPPOTrain(PPOTrain):
    @classmethod
    def load_train(
        cls, 
        policy_train_state: TrainState, 
        value_head_train_state: TrainState, 
        policy_model: FlaxPreTrainedModel, 
        value_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
    ) -> GPTJPPOTrain:
        mesh = policy_model.config.mesh
        assert mesh is not None
        assert mesh == value_head_model.config.mesh
        policy_train_state_partition_spec = match_partition_rules(policy_model.config.get_partition_rules(), policy_train_state)
        value_head_train_state_partition_spec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_train_state)

        @partial(
            pjit, 
            donate_argnums=(0, 1), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_train_state_partition_spec), 
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
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS('dp', None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS('dp', None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS('dp', None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS('dp', None))
            old_logprobs = with_named_sharding_constraint(old_logprobs, mesh, PS('dp', None))
            old_values = with_named_sharding_constraint(old_values, mesh, PS('dp', None))
            old_advantages = with_named_sharding_constraint(old_advantages, mesh, PS('dp', None))
            old_returns = with_named_sharding_constraint(old_returns, mesh, PS('dp', None))
            
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
                    rngs={'dropout': new_key} if new_key is not None else None, 
                )[:, :-1]
                values = jnp.squeeze(values, axis=-1)

                logits = model_output.logits
                logits = logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(float('-inf'))
                logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])
                logprobs = jnp.where(attention_mask[:, 1:] == 1, logprobs, 0.0)

                loss, info = loss_fn(
                    attention_mask[:, 1:], 
                    logprobs, 
                    values, 
                    should_take_action, 
                    old_logprobs, 
                    old_values, 
                    old_advantages, 
                    old_returns, 
                )
                return loss, info
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
        initial_policy_params: Optional[PyTree], 
        policy_params: PyTree, 
        value_head_params: PyTree, 
        initial_policy_model: Optional[FlaxPreTrainedModel], 
        policy_model: FlaxPreTrainedModel, 
        value_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Optional[Callable], 
        dp_shard_logits: bool=True, 
    ) -> GPTJPPOInference:
        mesh = policy_model.config.mesh
        assert mesh is not None
        assert mesh == value_head_model.config.mesh
        assert (initial_policy_params is None and initial_policy_model) is None or (initial_policy_params is not None and initial_policy_model is not None)
        has_initial_policy = initial_policy_params is not None
        initial_policy_params_partition_spec = None
        if has_initial_policy:
            initial_policy_params_partition_spec = match_partition_rules(initial_policy_model.config.get_partition_rules(), initial_policy_params)
        policy_params_partition_spec = match_partition_rules(initial_policy_model.config.get_partition_rules(), policy_params)
        value_head_params_partition_spec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_params)

        @partial(
            jax.jit, 
            static_argnames=('initial_policy_output_attentions', 'initial_policy_output_hidden_states', 'policy_output_attentions', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), initial_policy_params_partition_spec) if has_initial_policy is None else NamedSharding(mesh, PS()), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=PPOForwardOutput(
                initial_policy_raw_output=FlaxCausalLMOutput(
                    logits=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                ) if has_initial_policy else NamedSharding(mesh, PS()), 
                policy_raw_output=FlaxCausalLMOutput(
                    logits=NamedSharding(mesh, PS("dp", None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                ), 
                values=NamedSharding(mesh, PS()), 
            ), 
        )
        def _forward(
            initial_policy_params: Optional[PyTree], 
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
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))
            
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            initial_model_output = None
            if has_initial_policy:
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
                # trunc padded logits
                initial_model_output = initial_model_output.replace(logits=initial_model_output.logits.at[:, :, initial_policy_model.config.unpadded_vocab_size:].set(-float('inf')))
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
            # trunc padded logits
            model_output = model_output.replace(logits=model_output.logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(-float('inf')))

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            values = value_head_model.apply(
                {'params': value_head_params}, 
                model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if new_key is not None else None, 
            )
            values = jnp.squeeze(values, axis=-1)

            # assert sharding on outputs
            if dp_shard_logits:
                if has_initial_policy:
                    initial_model_output = initial_model_output.replace(logits=with_named_sharding_constraint(initial_model_output.logits, mesh, PS("dp", None, None)))
                model_output = model_output.replace(logits=with_named_sharding_constraint(model_output.logits, mesh, PS("dp", None, None)))
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
            assert loss_fn is not None, "loss_fn must be set to use eval_loss"
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS("dp", None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS("dp", None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS("dp", None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS("dp", None))
            old_logprobs = with_named_sharding_constraint(old_logprobs, mesh, PS("dp", None))
            old_values = with_named_sharding_constraint(old_values, mesh, PS("dp", None))
            old_advantages = with_named_sharding_constraint(old_advantages, mesh, PS("dp", None))
            old_returns = with_named_sharding_constraint(old_returns, mesh, PS("dp", None))
            
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
                rngs={'dropout': new_key} if new_key is not None else None, 
            )[:, :-1]
            values = jnp.squeeze(values, axis=-1)

            logits = model_output.logits
            logits = logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(float('-inf'))
            logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])
            logprobs = jnp.where(attention_mask[:, 1:] == 1, logprobs, 0.0)

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

class GPTJPolicy(PPOPolicy):
    def __init__(
        self, 
        inference: GPTJInference, 
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
    
    def act(self, text_history: TextHistory) -> TextHistory:
        
        raw_input_str = self.in_str_process(text_history_to_str(text_history))

        new_key = None
        if self.prng_key is not None:
            self.prng_key, new_key = jax.random.split(self.prng_key)
        model_outputs = self.inference.generate_from_str(
            input_strs=[raw_input_str], 
            prng_key=new_key, 
            blocking_strategy=self.blocking_strategy, 
            generation_config=self.generation_config, 
            input_token_process=self.input_token_process, 
            target_token_process=self.target_token_process, 
            trace=self.trace, 
        )

        raw_output_str = model_outputs.output_strs[0]
        output_str = raw_output_str.removeprefix(raw_input_str)
        output_str = self.out_str_process(output_str)

        return text_history+(Text(output_str, True),)
    
    def set_params(self, policy_params: PyTree) -> None:
        self.inference = self.inference.replace(params=policy_params)