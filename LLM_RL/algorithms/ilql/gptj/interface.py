from typing import Optional, Callable, Tuple
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.ilql.base_interface import ILQLTrain
from flax.training.train_state import TrainState
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from functools import partial
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS

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
            static_argnames=('detach_q1', 'detach_q2', 'detach_v', 'train'), 
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

            next_token_ids: jax.Array, 
            next_tokens_attention_mask: jax.Array, 
            next_tokens_position_ids: jax.Array, 
            next_dones: jax.Array, 

            prng_key: Optional[jax.random.PRNGKeyArray], 

            detach_q1: bool, 
            detach_q2: bool, 
            detach_v: bool, 

            train: bool=True, 
        ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS('dp', None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS('dp', None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS('dp', None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS('dp', None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS('dp', None))
            dones = with_named_sharding_constraint(dones, mesh, PS('dp'))
            next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS('dp', None))
            next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS('dp', None))
            next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS('dp', None))
            next_dones = with_named_sharding_constraint(next_dones, mesh, PS('dp'))

            # define loss function

            def grad_loss(base_params: PyTree, q1_head_params: PyTree, q2_head_params: PyTree, v_head_params: PyTree):

                new_key = None
                if new_key is not None:
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
                    if new_key is not None:
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
                
                


