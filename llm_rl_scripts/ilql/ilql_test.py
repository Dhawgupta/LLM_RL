from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask, create_path, get_enabled_save_path
import os
import optax
from JaxSeq.models.gptj.interface import GPTJTrain, GPTJInference
from JaxSeq.models.gptj.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from LLM_RL.algorithms.ilql.base_interface import ilql_loss, ILQLInferenceFull, ILQLTrain
from JaxSeq.generation_eval import generate_language, compute_metrics
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import TextEnv, TextHistory, Text, interact_environment, text_env_eval, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain
from LLM_RL.algorithms.ilql.gptj.interface import GPTJPolicy, GPTJInferenceFull, GPTJILQLTrain
from LLM_RL.heads.mlp_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.mlp_head import MLPHeadConfig
from JaxSeq.shard_model import shard_params_from_params
from flax.training.train_state import TrainState
from LLM_RL.algorithms.ilql.data import ILQLDataset, ILQLIterableDataset
from LLM_RL.utils import get_tensor_stats_np
from functools import partial
import numpy as np
from JaxSeq.logs import label_logs, log, pull_logs
import json
from llm_rl_scripts.ppo.ppo_test import BitsTestEnv
from LLM_RL.heads.mlp_head import load_train_state as load_head_train_state, ModelLoadMode as HeadModelLoadMode
from JaxSeq.train import train_loop
from LLM_RL.algorithms.ilql.data import ILQLData, ILQLDataset, ILQLIterableDataset

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    n_rounds: int=1, 
    epochs: int=1, 
    max_steps: Optional[int]=None, 
    
    lr: float=1e-5, 
    weight_decay: float=0.0, 

    train_bsize: int=32, 
    grad_accum_steps: int=1, 

    gradient_checkpoint: bool=False, 
    fsdp: bool=False, 

    max_length: int=512, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=None, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 

    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, model_mesh_shape), ('dp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    with open(convert_path(train_data_path), 'r') as f:
        raw_data = jsonl_load(f)
    
    ilql_data = []
    for item in raw_data:
        text_trajectory_chain = TextTrajectoryChain(
            text_trajectory=TextTrajectory(
                text_history=[
                    Text('<|endoftext|>', False), 
                    Text(item['out_text'], True), 
                ], 
                reward=[0.0, item['reward']], 
                done=True, 
            ), 
            next=None, 
        )
        token_trajectory_chain = TokenTrajectoryChain.from_text_trajectory_chain(text_trajectory_chain, tokenizer)
        ilql_data.append(ILQLData.from_token_trajectory_chain(token_trajectory_chain))
    
    dataset = ILQLDataset.from_ilql_data_list(
        ilql_data, 
        tokenizer, 
        BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_length, 
        )
    )

    def policy_optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
    
    def value_head_optim_getter(params: PyTree):
        mask = get_weight_decay_mask(("bias",))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )

    base_train_state, base_model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.float32, 
        optim_getter=policy_optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        force_pad_embeddings=force_pad_embeddings, 
        gradient_checkpoint=gradient_checkpoint, 
        fsdp=fsdp, 
        params_dtype=jnp.float32, 
    )
    with jax.default_device(jax.devices('cpu')[0]):
        target_base_params = jax.tree_util.tree_map(
            lambda x: x.copy(), 
            base_train_state.params, 
        )
    target_base_params = shard_params_from_params(
        model=base_model, 
        params=target_base_params, 
    )

    q1_head_train_state, q_head = load_head_train_state_from_config(
        model_config=MLPHeadConfig(
            input_dim=base_model.config.n_embd, 
            hidden_dim=base_model.config.n_embd, 
            output_dim=base_model.config.vocab_size, 
            use_bias=True, 
            initializer_range=0.0, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        pad_to_output_dim=None, 
        fsdp=fsdp, 
        params_dtype=jnp.float32, 
    )
    with jax.default_device(jax.devices('cpu')[0]):
        q1_target_head_params = jax.tree_util.tree_map(
            lambda x: x.copy(), 
            q1_head_train_state.params, 
        )
    q1_target_head_params = shard_params_from_params(
        model=q_head, 
        params=q1_target_head_params, 
    )

    # TODO: add rng key for loading
    q2_head_train_state, _ = load_head_train_state_from_config(
        model_config=MLPHeadConfig(
            input_dim=base_model.config.n_embd, 
            hidden_dim=base_model.config.n_embd, 
            output_dim=base_model.config.vocab_size, 
            use_bias=True, 
            initializer_range=0.0, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        pad_to_output_dim=None, 
        fsdp=fsdp, 
        params_dtype=jnp.float32, 
    )
    with jax.default_device(jax.devices('cpu')[0]):
        q2_target_head_params = jax.tree_util.tree_map(
            lambda x: x.copy(), 
            q2_head_train_state.params, 
        )
    q2_target_head_params = shard_params_from_params(
        model=q_head, 
        params=q2_target_head_params, 
    )

    v_head_train_state, v_head = load_head_train_state_from_config(
        model_config=MLPHeadConfig(
            input_dim=base_model.config.n_embd, 
            hidden_dim=base_model.config.n_embd, 
            output_dim=1, 
            use_bias=True, 
            initializer_range=0.0, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        pad_to_output_dim=None, 
        fsdp=fsdp, 
        params_dtype=jnp.float32, 
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    loss_fn = partial(ilql_loss, gamma=0.99, tau=0.7, cql_weight=0.1)

    train = GPTJILQLTrain.load_train(
        base_train_state=base_train_state, 
        target_base_params=target_base_params, 
        q1_head_train_state=q1_head_train_state, 
        q2_head_train_state=q2_head_train_state, 
        v_head_train_state=v_head_train_state, 
        q1_target_head_params=q1_target_head_params, 
        q2_target_head_params=q2_target_head_params, 
        base_model=base_model, 
        q_head_model=q_head, 
        v_head_model=v_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn, 
        detach_q1=False, 
        detach_q2=False, 
        detach_v=False, 
        polyak_alpha=0.005, 
        hard_update_every=None, 
    )
    
    env = BitsTestEnv(n=10)
    
    # policy_prng = jax.random.PRNGKey(0)
    # policy = GPTJPolicy(
    #     inference=policy_inference, 
    #     prng_key=policy_prng, 
    #     generation_config=GenerationConfig(
    #         do_sample=policy_do_sample, 
    #         num_beams=policy_num_beams, 
    #         temperature=policy_temperature, 
    #         top_p=policy_top_p, 
    #         top_k=policy_top_k, 
    #         eos_token_id=tokenizer.encode('\n')[0], 
    #         pad_token_id=tokenizer.pad_token_id, 
    #         max_new_tokens=max_output_length, 
    #     ), 
    #     blocking_strategy=BlockingStrategy(
    #         padding=Padding.LEFT, 
    #         truncation=Truncation.LEFT, 
    #         max_length=max_input_length, 
    #     ), 
    #     out_str_process=lambda x: x.removesuffix('\n')+'\n', 
    # )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )
    
    train_prng = jax.random.PRNGKey(1)
    ppo_trainer, ppo_inference, policy = train_loop(
        trainer=ppo_trainer, 
        inference=ppo_inference, 
        evaluator=None, 
        dataset=dataset, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        n_rounds=n_rounds, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
