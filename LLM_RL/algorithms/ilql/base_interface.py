from __future__ import annotations
from typing import Union, Tuple, Any, Callable, Optional
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
    is_next_state = is_next_state.at[jnp.arange(0, is_next_state.shape[0], dtype=jnp.int32), jnp.argmax(is_next_state.astype(jnp.int32), axis=1)].set(False)
    is_next_state = jnp.concatenate((is_next_state, (should_take_action.sum(axis=1) > 0)[..., None]), axis=1)

    vns_query_indicators = get_query_indicators(is_next_state.reshape(-1))
    vns_query_indicators = vns_query_indicators[:qv_query_indicators.shape[0], :] # TODO: check this
    
    q1sa_selected = (qv_query_indicators * q1sa_flat).sum(axis=1)
    q2sa_selected = (qv_query_indicators * q2sa_flat).sum(axis=1)
    v_selected = (qv_query_indicators * v_flat).sum(axis=1)
    target_q1sa_selected = (qv_query_indicators * target_q1sa_flat).sum(axis=1)
    target_q2sa_selected = (qv_query_indicators * target_q2sa_flat).sum(axis=1)
    vns_selected = (vns_query_indicators * vns_flat).sum(axis=1)
    rs_selected = (qv_query_indicators * rewards.reshape(-1)).sum(axis=1)

    sa_mask = (qv_query_indicators.sum(axis=1) > 0).astype(jnp.float32)
    ns_mask = (vns_query_indicators.sum(axis=1) > 0).astype(jnp.float32)

    n_sa, n_ns = sa_mask.sum(), ns_mask.sum()
    assert n == n_sa and n == n_ns

    q1_loss = (optax.l2_loss(q1sa_selected, jax.lax.stop_gradient(rs_selected + gamma * vns_selected)) * sa_mask).sum() / n
    q2_loss = (optax.l2_loss(q2sa_selected, jax.lax.stop_gradient(rs_selected + gamma * vns_selected)) * sa_mask).sum() / n

    target_q_selected = jnp.minimum(target_q1sa_selected, target_q2sa_selected)
    expectile_indicator = (target_q_selected >= v_selected).astype(jnp.float32)
    expectile_weights = expectile_indicator * tau + (1 - expectile_indicator) * (1 - tau)
    v_loss = (optax.l2_loss(v_selected, jax.lax.stop_gradient(target_q_selected)) * jax.lax.stop_gradient(expectile_weights) * sa_mask).sum() / n

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
        q1=get_tensor_stats(q1sa_selected, mask=sa_mask), 
        q2=get_tensor_stats(q2sa_selected, mask=sa_mask), 
        v=get_tensor_stats(v_selected, mask=sa_mask), 
        target_q=get_tensor_stats(target_q_selected, mask=sa_mask), 
        target_q1=get_tensor_stats(target_q1sa_selected, mask=sa_mask), 
        target_q2=get_tensor_stats(target_q2sa_selected, mask=sa_mask), 
        vns=get_tensor_stats(vns_selected, mask=ns_mask), 
        final_v=get_tensor_stats(v_final, mask=ns_mask), 
        rewards=get_tensor_stats(rewards, mask=sa_mask), 
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

    #     next_token_ids: jax.Array, 
    #     next_tokens_attention_mask: jax.Array, 
    #     next_tokens_position_ids: jax.Array, 
    #     next_dones: jax.Array, 

    #     prng_key: Optional[jax.random.PRNGKeyArray], 

    #     detach_q1: bool, 
    #     detach_q2: bool, 
    #     detach_v: bool, 

    #     train: bool=True, 
    # ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        rewards: jax.Array, # [batch, time-1]
        dones: jax.Array, # [batch]
        next_token_ids: jax.Array, # [batch, n_time]
        next_dones: jax.Array, # [batch]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        next_tokens_attention_mask: Optional[jax.Array]=None, 
        next_tokens_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
        detach_q1: bool=False, 
        detach_q2: bool=False, 
        detach_v: bool=False, 
    ) -> Tuple[ILQLTrain, jax.Array, PyTree]:
        
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )
        
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
            detach_q1=detach_q1, 
            detach_q2=detach_q2, 
            detach_v=detach_v, 
            train=train, 
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

