from __future__ import annotations
from typing import Union, Tuple, Any, Callable, Optional, NamedTuple, List
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray
import optax
from LLM_RL.utils import get_tensor_stats
from flax import struct
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from jaxtyping import PyTree
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from flax.core import freeze
from transformers.generation import GenerationConfig
import numpy as np
from JaxSeq.utils import block_sequences, BlockingStrategy, Padding, Truncation
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from JaxSeq.models.base_interface import GenerationFromStrOutput, Inference
from LLM_RL.environment import BatchedTextPolicy

# loss function

def get_query_indicators(
    flat_mask: jax.Array, 
) -> jax.Array:
    idxs = jnp.argwhere(flat_mask, size=flat_mask.shape[0], fill_value=flat_mask.shape[0])[:, 0]
    query_indicators = jax.nn.one_hot(idxs, num_classes=flat_mask.shape[0]+1, dtype=jnp.float32)[:, :-1]
    return query_indicators

def ilql_loss(
    q1: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    q2: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    v: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    v_final: jax.Array, # [batch]
    target_q1: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    target_q2: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    q1_logits: jax.Array, # [batch, time-1, vocab] output is masked; shift x[:-1]
    q2_logits: jax.Array, # [batch, time-1, vocab] output is masked; shift x[:-1]
    token_ids: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    attention_mask: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    should_take_action: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    rewards: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    *, 
    gamma: Union[float, jax.Array], 
    tau: Union[float, jax.Array], 
    cql_weight: Union[float, jax.Array], 
) -> Tuple[jnp.ndarray, Any]:
    # should be an action in the batch
    mask = should_take_action.astype(jnp.float32) * attention_mask
    n = mask.sum()
    
    q1sa_flat, q2sa_flat, v_flat = q1.reshape(-1), q2.reshape(-1), v.reshape(-1)
    target_q1sa_flat, target_q2sa_flat = target_q1.reshape(-1), target_q2.reshape(-1)
    vns_flat = jnp.concatenate((v, v_final[..., None]), axis=1).reshape(-1)

    qv_query_indicators = get_query_indicators(should_take_action.reshape(-1))

    is_next_state = should_take_action.copy()
    # set first action position to false
    is_next_state = is_next_state.at[jnp.arange(0, is_next_state.shape[0], dtype=jnp.int32), jnp.argmax(is_next_state.astype(jnp.int32), axis=1)].set(False)
    # set endpoint to true as long as there is at least 1 action in the sequence
    is_next_state = jnp.concatenate((is_next_state, (should_take_action.sum(axis=1) > 0)[..., None]), axis=1)

    vns_query_indicators = get_query_indicators(is_next_state.reshape(-1))
    # should be the same number of vns as qv, so we can clip the extra padding to match shape
    vns_query_indicators = vns_query_indicators[:qv_query_indicators.shape[0], :]
    
    # extract selected values
    q1sa_selected = (qv_query_indicators * q1sa_flat).sum(axis=1)
    q2sa_selected = (qv_query_indicators * q2sa_flat).sum(axis=1)
    v_selected = (qv_query_indicators * v_flat).sum(axis=1)
    target_q1sa_selected = (qv_query_indicators * target_q1sa_flat).sum(axis=1)
    target_q2sa_selected = (qv_query_indicators * target_q2sa_flat).sum(axis=1)
    vns_selected = (vns_query_indicators * vns_flat).sum(axis=1)
    rs_selected = (qv_query_indicators * rewards.reshape(-1)).sum(axis=1)

    # get masks for selected values
    sa_mask = (qv_query_indicators.sum(axis=1) > 0).astype(jnp.float32)
    ns_mask = (vns_query_indicators.sum(axis=1) > 0).astype(jnp.float32)

    # compute q loss
    q1_loss = (optax.l2_loss(q1sa_selected, jax.lax.stop_gradient(rs_selected + gamma * vns_selected)) * sa_mask).sum() / n
    q2_loss = (optax.l2_loss(q2sa_selected, jax.lax.stop_gradient(rs_selected + gamma * vns_selected)) * sa_mask).sum() / n

    # compute v loss
    target_q_selected = jnp.minimum(target_q1sa_selected, target_q2sa_selected)
    expectile_indicator = (target_q_selected >= v_selected).astype(jnp.float32)
    expectile_weights = expectile_indicator * tau + (1 - expectile_indicator) * (1 - tau)
    v_loss = (optax.l2_loss(v_selected, jax.lax.stop_gradient(target_q_selected)) * jax.lax.stop_gradient(expectile_weights) * sa_mask).sum() / n

    # compute cql loss on both q heads
    q1_cql_loss = optax.softmax_cross_entropy_with_integer_labels(q1_logits, token_ids)
    q1_cql_loss = (mask * q1_cql_loss).sum() / n

    q2_cql_loss = optax.softmax_cross_entropy_with_integer_labels(q2_logits, token_ids)
    q2_cql_loss = (mask * q2_cql_loss).sum() / n
    
    loss = q1_loss + q2_loss + v_loss + cql_weight * (q1_cql_loss + q2_cql_loss)

    logs = dict(
        losses=dict(
            total_loss=loss, 
            q1_loss=q1_loss, 
            q2_loss=q2_loss, 
            v_loss=v_loss, 
            q1_cql_loss=q1_cql_loss, 
            q2_cql_loss=q2_cql_loss, 
        ), 
        q1=get_tensor_stats(q1sa_selected, mask=sa_mask, n=n), 
        q2=get_tensor_stats(q2sa_selected, mask=sa_mask, n=n), 
        v=get_tensor_stats(v_selected, mask=sa_mask, n=n), 
        target_q=get_tensor_stats(target_q_selected, mask=sa_mask, n=n), 
        target_q1=get_tensor_stats(target_q1sa_selected, mask=sa_mask, n=n), 
        target_q2=get_tensor_stats(target_q2sa_selected, mask=sa_mask, n=n), 
        vns=get_tensor_stats(vns_selected, mask=ns_mask, n=n), 
        v_final=get_tensor_stats(v_final, mask=jnp.ones(v_final.shape, dtype=jnp.int32), n=v_final.shape[0]), 
        rewards=get_tensor_stats(rewards, mask=mask, n=n), 
    )

    return loss, logs

class ILQLTrain(struct.PyTreeNode):
    base_train_state: TrainState
    target_base_params: Optional[PyTree]
    q1_head_train_state: TrainState
    q2_head_train_state: TrainState
    v_head_train_state: TrainState
    q1_target_head_params: PyTree
    q2_target_head_params: PyTree
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    v_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     base_train_state: TrainState, 
    #     target_base_params: Optional[PyTree], 
    #     q1_head_train_state: TrainState, 
    #     q2_head_train_state: TrainState, 
    #     v_head_train_state: TrainState, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 

    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     rewards: jax.Array, 
    #     dones: jax.Array, 

    #     next_token_ids: Optional[jax.Array], 
    #     next_tokens_attention_mask: Optional[jax.Array], 
    #     next_tokens_position_ids: Optional[jax.Array], 
    #     next_dones: Optional[jax.Array], 

    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        rewards: jax.Array, # [batch, time-1]
        dones: jax.Array, # [batch]
        next_token_ids: Optional[jax.Array], # [batch, n_time]
        next_dones: Optional[jax.Array], # [batch]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        next_tokens_attention_mask: Optional[jax.Array]=None, 
        next_tokens_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[ILQLTrain, jax.Array, PyTree]:
        
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        if next_token_ids is not None:
            next_tokens_attention_mask, next_tokens_position_ids = initialize_attn_mask_pos_ids(
                next_token_ids, 
                self.tokenizer.pad_token_id, 
                next_tokens_attention_mask, 
                next_tokens_position_ids, 
            )
        else:
            assert next_tokens_attention_mask is None
            assert next_tokens_position_ids is None
        
        base_train_state, \
        target_base_params, \
        q1_head_train_state, \
        q2_head_train_state, \
        v_head_train_state, \
        q1_target_head_params, \
        q2_target_head_params, \
        loss, logs = self._step(
            self.base_train_state, 
            self.target_base_params, 
            self.q1_head_train_state, 
            self.q2_head_train_state, 
            self.v_head_train_state, 
            self.q1_target_head_params, 
            self.q2_target_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            rewards, 
            dones, 
            next_token_ids, 
            next_tokens_attention_mask, 
            next_tokens_position_ids, 
            next_dones, 
            prng_key, 
            train, 
        )

        return self.replace(
            base_train_state=base_train_state, 
            target_base_params=target_base_params, 
            q1_head_train_state=q1_head_train_state, 
            q2_head_train_state=q2_head_train_state, 
            v_head_train_state=v_head_train_state, 
            q1_target_head_params=q1_target_head_params, 
            q2_target_head_params=q2_target_head_params, 
        ), loss, logs

class ILQLSimpleForwardOutput(NamedTuple):
    base_raw_output: FlaxCausalLMOutput
    q1: jax.Array
    q2: jax.Array
    v: jax.Array

class ILQLInferenceSimple(struct.PyTreeNode):
    pi_beta_params: Optional[PyTree]
    base_params: PyTree
    q1_head_params: PyTree
    q2_head_params: PyTree
    v_head_params: PyTree
    pi_beta_model: Optional[FlaxPreTrainedModel] = struct.field(pytree_node=False)
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    v_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _generate: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)

    # def _generate(
    #     pi_beta_params: PyTree, 
    #     base_params: PyTree, 
    #     q1_head_params: PyTree, 
    #     q2_head_params: PyTree, 
    #     v_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     generation_config: Optional[FrozenDict]=None, 
    #     trace: bool=True, 
    # ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]
    
    # def _forward(
    #     base_params: PyTree, 
    #     q1_head_params: PyTree, 
    #     q2_head_params: PyTree, 
    #     v_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     output_attentions: Optional[bool]=None, 
    #     train: bool=False, 
    # ) -> ILQLSimpleForwardOutput:
    #     raise NotImplementedError

    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        if self.pi_beta_params is None:
            raise NotImplementedError
        
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._generate(
            self.pi_beta_params, 
            self.base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            self.v_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )
    
    def generate_from_str(
        self, 
        input_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        generation_config: Optional[GenerationConfig]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ) -> GenerationFromStrOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # generate
        outputs = self.generate(
            jnp.asarray(tokens), 
            prng_key, 
            generation_config=generation_config, 
            trace=trace
        )
        # process outputs
        output_sequences = list(map(target_token_process, outputs.sequences.tolist()))
        output_scores = None
        if isinstance(outputs, FlaxBeamSearchOutput):
            output_scores = np.asarray(outputs.scores)
        # decode tokens
        output_strs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return GenerationFromStrOutput(output_strs, output_scores)
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> ILQLSimpleForwardOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            self.v_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            output_attentions, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxCausalLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs

class ILQLFullForwardOutput(NamedTuple):
    base_raw_output: FlaxCausalLMOutput
    target_base_raw_output: Optional[FlaxCausalLMOutput]
    q1: jax.Array
    q2: jax.Array
    v: jax.Array
    q1_target: jax.Array
    q2_target: jax.Array

class ILQLInferenceFull(struct.PyTreeNode):
    pi_beta_params: Optional[PyTree]
    base_params: PyTree
    target_base_params: Optional[PyTree]
    q1_head_params: PyTree
    q2_head_params: PyTree
    v_head_params: PyTree
    q1_target_head_params: PyTree
    q2_target_head_params: PyTree
    pi_beta_model: Optional[FlaxPreTrainedModel] = struct.field(pytree_node=False)
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    v_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _generate: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _eval_loss: Callable = struct.field(pytree_node=False)

    # def _generate(
    #     pi_beta_params: PyTree, 
    #     base_params: PyTree, 
    #     target_base_params: Optional[PyTree], 
    #     v_head_params: PyTree, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     generation_config: Optional[FrozenDict]=None, 
    #     trace: bool=True, 
    # )
    
    # def _forward(
    #     base_params: PyTree, 
    #     target_base_params: Optional[PyTree], 
    #     q1_head_params: PyTree, 
    #     q2_head_params: PyTree, 
    #     v_head_params: PyTree, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     output_attentions: Optional[bool]=None, 
    #     train: bool=False, 
    # ) -> ILQLForwardOutput:
    #     raise NotImplementedError

    # def _eval_loss(
    #     base_params: PyTree, 
    #     target_base_params: Optional[PyTree], 
    #     q1_head_params: PyTree, 
    #     q2_head_params: PyTree, 
    #     v_head_params: PyTree, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 

    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     rewards: jax.Array, 
    #     dones: jax.Array, 

    #     next_token_ids: Optional[jax.Array], 
    #     next_tokens_attention_mask: Optional[jax.Array], 
    #     next_tokens_position_ids: Optional[jax.Array], 
    #     next_dones: Optional[jax.Array], 

    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError

    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        if self.pi_beta_params is None:
            raise NotImplementedError
        
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._generate(
            self.pi_beta_params, 
            self.base_params, 
            self.target_base_params, 
            self.v_head_params, 
            self.q1_target_head_params, 
            self.q2_target_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )
    
    def generate_from_str(
        self, 
        input_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        generation_config: Optional[GenerationConfig]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ) -> GenerationFromStrOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # generate
        outputs = self.generate(
            jnp.asarray(tokens), 
            prng_key, 
            generation_config=generation_config, 
            trace=trace
        )
        # process outputs
        output_sequences = list(map(target_token_process, outputs.sequences.tolist()))
        output_scores = None
        if isinstance(outputs, FlaxBeamSearchOutput):
            output_scores = np.asarray(outputs.scores)
        # decode tokens
        output_strs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return GenerationFromStrOutput(output_strs, output_scores)
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> ILQLFullForwardOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.base_params, 
            self.target_base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            self.v_head_params, 
            self.q1_target_head_params, 
            self.q2_target_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            output_attentions, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxCausalLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs
    
    def eval_loss(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        rewards: jax.Array, # [batch, time-1]
        dones: jax.Array, # [batch]
        next_token_ids: Optional[jax.Array], # [batch, n_time]
        next_dones: Optional[jax.Array], # [batch]
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        next_tokens_attention_mask: Optional[jax.Array]=None, 
        next_tokens_position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=True, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )

        if next_token_ids is not None:
            next_tokens_attention_mask, next_tokens_position_ids = initialize_attn_mask_pos_ids(
                next_token_ids, 
                self.tokenizer.pad_token_id, 
                next_tokens_attention_mask, 
                next_tokens_position_ids, 
            )
        else:
            assert next_tokens_attention_mask is None
            assert next_tokens_position_ids is None
        
        loss, logs = self._eval_loss(
            self.base_params, 
            self.target_base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            self.v_head_params, 
            self.q1_target_head_params, 
            self.q2_target_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            rewards, 
            dones, 
            next_token_ids, 
            next_tokens_attention_mask, 
            next_tokens_position_ids, 
            next_dones, 
            prng_key, 
            train, 
        )

        return loss, logs
    
    def eval_loss_from_str(self, *args, **kwargs):
        raise NotImplementedError

class ILQLPolicy(BatchedTextPolicy):
    def set_params(self, policy_params: PyTree) -> None:
        raise NotImplementedError
