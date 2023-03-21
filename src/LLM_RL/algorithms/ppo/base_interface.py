from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional, Union, Tuple, Callable, NamedTuple, Dict, Any
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, BlockingStrategy, block_sequences, Padding, Truncation
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from flax.core import FrozenDict, freeze
import flax.linen as nn
from LLM_RL.utils import get_tensor_stats
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from LLM_RL.environment import TextTrajectoryChain, text_history_to_str, text_trajectory_chain_to_token_trajectory_chain

def ppo_loss_fn(
    attention_mask: jax.Array, # [batch, time]
    logprobs: jax.Array, # [batch, time-1]
    values: jax.Array, # [batch, time]
    is_actions: jax.Array, # [batch, time]
    old_logprobs: jax.Array, # [batch, time-1]
    old_values: jax.Array, # [batch, time]
    old_advantages: jax.Array, # [batch, time-1]
    old_returns: jax.Array, # [batch, time]
    *, 
    cliprange_value: Union[float, jax.Array], 
    cliprange: Union[float, jax.Array], 
    value_loss_weight: Union[float, jax.Array], 
) -> Tuple[jax.Array, Dict[str, Any]]:
    """PPO objective function.
    References:
    - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
    - https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
    """
    mask = is_actions[:, 1:].astype(jnp.float32) * attention_mask[:, 1:]
    n = mask.sum()
    
    values_clipped = jnp.clip(
        values, 
        old_values - cliprange_value, 
        old_values + cliprange_value, 
    )

    vf_loss1 = (values - old_returns) ** 2
    vf_loss2 = (values_clipped - old_returns) ** 2
    vf_loss = 0.5 * jnp.sum(jnp.maximum(vf_loss1, vf_loss2)[:, :-1] * mask) / n
    vf_clipfrac = jnp.sum((vf_loss2 > vf_loss1).astype(jnp.float32)[:, :-1] * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = jnp.exp(log_ratio)
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    approx_kl = jnp.mean((ratio - 1) - log_ratio)

    pg_loss1 = -old_advantages * ratio
    pg_loss2 = -old_advantages * jnp.clip(
        ratio, 
        1.0 - cliprange, 
        1.0 + cliprange, 
    )
    pg_loss = jnp.sum(jnp.maximum(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = jnp.sum((pg_loss2 > pg_loss1).astype(jnp.float32) * mask) / n

    loss = pg_loss + value_loss_weight * vf_loss

    logs = dict(
        losses=dict(
            total_loss=loss, 
            policy_loss=pg_loss, 
            value_loss=vf_loss, 
        ), 
        values=dict(
            get_tensor_stats(values[:, :-1], mask, n), 
            values_error=jnp.sum(((values - old_returns)[:, :-1] * mask) ** 2) / n, 
            clipfrac=vf_clipfrac, 
        ), 
        old_values=get_tensor_stats(old_values[:, :-1], mask, n), 
        returns=get_tensor_stats(old_returns[:, :-1], mask, n), 
        policy=dict(
            approx_kl=approx_kl, 
            clipfrac=pg_clipfrac, 
        ), 
        ratio=(ratio * mask).sum() / n, 
        padding_percentage=n / mask.size, 
    )

    return loss, logs

class PPOForwardOutput(NamedTuple):
    initial_policy_raw_output: FlaxCausalLMOutput
    policy_raw_output: FlaxCausalLMOutput
    values: jax.Array

class PPOData(NamedTuple):
    input_ids: jax.Array
    attention_mask: jax.Array
    position_ids: jax.Array
    is_actions: jax.Array
    old_logprobs: jax.Array
    old_values: jax.Array
    old_advantages: jax.Array
    old_returns: jax.Array

class PPOInference(struct.PyTreeNode):
    initial_policy_params: PyTree
    policy_params: PyTree
    value_head_params: PyTree
    initial_base_model: FlaxPreTrainedModel = struct.field(pytree_node=False) # corresponds to initial_policy_params
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False) # corresponds to policy_params
    value_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _eval_loss: Optional[Callable] = struct.field(pytree_node=False, default=None)
    
    # def _forward(
    #     initial_policy_params: PyTree, 
    #     policy_params: PyTree, 
    #     value_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     initial_policy_output_attentions: Optional[bool]=None, 
    #     initial_policy_output_hidden_states: Optional[bool]=None, 
    #     policy_output_attentions: Optional[bool]=None, # no policy_output_hidden_states option because this is required
    #     train: bool=False, 
    # ) -> PPOForwardOutput:
    #     raise NotImplementedError

    # TODO: should I add other arguments like old_values, old_returns, old_advantages, old_logprobs, etc.?
    # def _eval_loss(
    #     initial_policy_params: PyTree, 
    #     policy_params: PyTree, 
    #     value_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        initial_policy_output_attentions: Optional[bool]=None, 
        initial_policy_output_hidden_states: Optional[bool]=None, 
        policy_output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> PPOForwardOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.initial_policy_params, 
            self.policy_params, 
            self.value_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            initial_policy_output_attentions, 
            initial_policy_output_hidden_states, 
            policy_output_attentions, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        initial_policy_output_attentions: Optional[bool]=None, 
        initial_policy_output_hidden_states: Optional[bool]=None, 
        policy_output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> PPOForwardOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            initial_policy_output_attentions=initial_policy_output_attentions, 
            initial_policy_output_hidden_states=initial_policy_output_hidden_states, 
            policy_output_attentions=policy_output_attentions, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs
    
    def get_training_data_from_text_trajectory_chain(
        self, 
        text_trajectory_chains: List[TextTrajectoryChain], 
        bsize: int, 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> List[PPOData]:
        
        token_trajectory_chains = [text_trajectory_chain_to_token_trajectory_chain(item, self.tokenizer) for item in text_trajectory_chains]

        # check truncation conditions
        tokens = []
        if blocking_strategy.max_length is not None:
            for token_trajectory_chain in token_trajectory_chains:
                curr_trajectory = token_trajectory_chain.token_trajectory
                while curr_trajectory is not None:
                    # check that the trajectory is not truncated or that it doesn't end with a state and start with an action
                    # we can't calculate the advantage if the trajectory is truncated and there are later actions
                    no_trunc = curr_trajectory.tokens.shape[0] <= blocking_strategy.max_length
                    ends_with_state = (not np.any(curr_trajectory.is_action[blocking_strategy.max_length:]))
                    next_starts_with_action = (curr_trajectory.next is None) or curr_trajectory.is_action[0]

                    assert not (ends_with_state and next_starts_with_action), 'trajectory truncation error'
                    assert no_trunc or ends_with_state, 'trajectory truncation error'

                    tokens.append(curr_trajectory.tokens)

                    curr_trajectory = curr_trajectory.next

        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)

        # get values, logits from forward pass
        forward_batch_outputs = []
        for i in range(0, len(tokens), bsize):
            forward_batch_output = self.forward(
                jnp.asarray(tokens[i:i+bsize], dtype=jnp.int32), 
                train=train, 
                prng_key=prng_key, 
            )
            forward_batch_outputs.append(forward_batch_output)
        
        initial_policy_logits = []
        policy_logits = []
        values = []

        # TODO:
        # unpad, logits values
        # finish data loading function
        # build eval_loss
        # test on some toy data
        # batched policy / environment abstractions
        # environment data conversion system
        # training loop

        for b_idx, forward_batch_output in enumerate(forward_batch_outputs):
            for idx in range(b_idx*bsize, (b_idx+1)*bsize):
                pad_t = np.where(tokens[idx] == self.tokenizer.pad_token_id)[0]
                if len(pad_t) > 0:
                    pass
                    
        
        # compute logprobs


        # all_input_strs = []
        # for item in text_trajectory_chain:
        #     chain_input_strs = []
        #     node = item
        #     while node is not None:
        #         chain_input_strs.append(text_history_to_str(node.text_trajectory.text_history))
        #         node = node.next
        #     all_input_strs.append(chain_input_strs)
    

    # def eval_loss(
    #     self, 
    #     input_ids: jax.Array, 
    #     target_ids: jax.Array, 
    #     input_attention_mask: Optional[jax.Array]=None, 
    #     input_position_ids: Optional[jax.Array]=None, 
    #     target_attention_mask: Optional[jax.Array]=None, 
    #     target_position_ids: Optional[jax.Array]=None, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
    #         input_ids, 
    #         self.tokenizer.pad_token_id, 
    #         input_attention_mask, 
    #         input_position_ids, 
    #     )
    #     target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
    #         target_ids, 
    #         self.tokenizer.pad_token_id, 
    #         target_attention_mask, 
    #         target_position_ids, 
    #     )

    #     return self._eval_loss(
    #         self.params, 
    #         input_ids, 
    #         input_attention_mask, 
    #         input_position_ids, 
    #         target_ids, 
    #         target_attention_mask, 
    #         target_position_ids, 
    #         prng_key, 
    #         train, 
    #     )
    
    # def eval_loss_from_str(
    #     self, 
    #     loss_fn: Callable, 
    #     input_strs: List[str], 
    #     target_strs: List[str], 
    #     input_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
    #     target_blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    #     input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    #     target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     if input_token_process is None:
    #         input_token_process = lambda x: x
    #     if target_token_process is None:
    #         target_token_process = lambda x: x
    #     # tokenize
    #     input_tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
    #     input_tokens = block_sequences(input_tokens, self.tokenizer.pad_token_id, np.int32, input_blocking_strategy)
    #     target_tokens = [target_token_process(self.tokenizer.encode(item)) for item in target_strs]
    #     target_tokens = block_sequences(target_tokens, self.tokenizer.pad_token_id, np.int32, target_blocking_strategy)
    #     # loss
    #     return self.eval_loss(
    #         loss_fn, 
    #         jnp.asarray(input_tokens), 
    #         jnp.asarray(target_tokens), 
    #         prng_key=prng_key, 
    #         train=train, 
    #     )

class PPOTrain(struct.PyTreeNode):
    policy_train_state: TrainState
    value_head_train_state: TrainState
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    value_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     policy_train_state: TrainState, 
    #     value_head_train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     is_actions: jax.Array, 
    #     old_logprobs: jax.Array, 
    #     old_values: jax.Array, 
    #     old_advantages: jax.Array, 
    #     old_returns: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, TrainState, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        is_actions: jax.Array, # [batch, time]
        old_logprobs: jax.Array, # [batch, time-1]
        old_values: jax.Array, # [batch, time]
        old_advantages: jax.Array, # [batch, time-1]
        old_returns: jax.Array, # [batch, time]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[PPOTrain, jax.Array, PyTree]:
        
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )
        
        policy_train_state, value_head_train_state, loss, logs = self._step(
            self.policy_train_state, 
            self.value_head_train_state, 
            input_ids, 
            attention_mask, 
            position_ids, 
            is_actions, 
            old_logprobs, 
            old_values, 
            old_advantages, 
            old_returns, 
            prng_key, 
            train, 
        )

        return self.replace(
            policy_train_state=policy_train_state, 
            value_head_train_state=value_head_train_state, 
        ), loss, logs
