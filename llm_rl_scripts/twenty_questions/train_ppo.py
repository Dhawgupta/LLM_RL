from typing import Optional, Dict, Any, Tuple
from jaxtyping import PyTree
from functools import partial
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
import optax
import pickle as pkl
import re
import tyro
import random
from flax.training.train_state import TrainState
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.data import Seq2SeqDataset
from JaxSeq.generation_eval import generate_language, compute_metrics
from JaxSeq.logs import label_logs, log, pull_logs
from JaxSeq.models.gpt2.interface import GPT2Train, GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
from JaxSeq.shard_model import shard_params_from_params
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask, create_path, get_enabled_save_path
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype, setup_experiment_save, multihost_device_get
from LLM_RL.algorithms.ppo.base_interface import ppo_loss_fn, FixedKLController, AdaptiveKLController
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2Policy, GPT2ILQLInference, GPT2PPOTrain
from LLM_RL.algorithms.ppo.train import train_loop
from LLM_RL.environment import TextEnv, TextHistory, Text, interact_environment, text_env_eval, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain, TokenTrajectory
from LLM_RL.heads.linear_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.linear_head import LinearHeadConfig
from LLM_RL.algorithms.ppo.data import PPODataset, PPOIterableDataset
from LLM_RL.utils import get_tensor_stats_np
from .env import TwentyQuestionsPolicyEnvironment
from .oracle import T5Oracle
from .oracle import T5ModelLoadMode as T5OracleModelLoadMode
from .data import get_default_word_list, create_conversation_from_history



def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    n_rounds: int=1, 
    epochs: int=1, 
    max_steps: Optional[int]=None, 

    seed: int=0,
    
    lr: float=1e-5, 
    weight_decay: float=0.0, 

    train_bsize: int=32, 
    grad_accum_steps: int=1, 
    rollout_bsize: int=32, 
    n_rollouts: int=128, 
    ppo_data_bsize: int=32, 

    env_deterministic: bool=False,
    oracle_model_mode: T5OracleModelLoadMode=T5OracleModelLoadMode.PARAMS,
    oracle_model_path: str="gcs://rail-tpus-charles-3/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1/model_2.pkl",

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 
    use_fp16_activations: bool=False,
    use_fp16_params: bool=False,

    max_input_length: int=512, 
    max_output_length: int=512, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=None, 
    eval_every_epochs: Optional[int]=None, 
    eval_every_rounds: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 

    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_every_rounds: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_ppo_dataset: bool=True, 
    save_bf16: bool=True, 

    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None, 

    gamma: float=1.0, 
    lam: float=0.95, 
    use_advantage_whitening: bool=True, 

    init_kl_coef: float=0.001, 
    kl_target: Optional[float]=None, 
    kl_horizon: Optional[int]=None, 

    cliprange_value: float=0.2, 
    cliprange: float=0.2, 
    value_loss_coef: float=1.0, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    input_args = locals().copy()
    print(input_args)

    prng_key = jax.random.PRNGKey(seed)

    use_adaptive_kl = (kl_target is not None and kl_horizon is not None)
    if not use_adaptive_kl:
        assert kl_target is None and kl_horizon is None

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

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

    model_dtype = get_dtype(use_fp16=use_fp16_activations)
    params_dtype = get_dtype(use_fp16=use_fp16_params)

    model_prng_key, prng_key = jax.random.split(prng_key)
    policy_train_state, policy_model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=model_dtype, 
        optim_getter=policy_optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=params_dtype, 
    )
    policy_model.config.gradient_checkpointing = gradient_checkpointing
    policy_model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    with jax.default_device(jax.devices('cpu')[0]):
        initial_policy_params = jax.tree_util.tree_map(
            lambda x: multihost_device_get(x, mesh=mesh).copy(), 
            policy_train_state.params, 
        )
    initial_policy_params = shard_params_from_params(
        model=policy_model, 
        params=initial_policy_params, 
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)

    policy_inference = GPT2Inference.load_inference(
        params=policy_train_state.params, 
        model=policy_model, 
        tokenizer=tokenizer, 
    )

    prng_key, policy_prng = jax.random.split(prng_key)
    policy = GPT2Policy(
        inference=policy_inference, 
        prng_key=policy_prng, 
        generation_config=GenerationConfig(
            do_sample=policy_do_sample, 
            num_beams=policy_num_beams, 
            temperature=policy_temperature, 
            top_p=policy_top_p, 
            top_k=policy_top_k, 
            eos_token_id=tokenizer.encode('\n')[0], 
            pad_token_id=tokenizer.pad_token_id, 
            max_new_tokens=max_output_length, 
        ), 
        blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length, 
        ), 
        out_str_process=lambda x: x.removesuffix('\n')+'\n', 
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
    
    head_prng_key, prng_key = jax.random.split(prng_key)
    value_head_train_state, value_head = load_head_train_state_from_config(
        model_config=LinearHeadConfig(
            input_dim=policy_model.config.n_embd, 
            output_dim=1, 
            use_bias=True, 
            initializer_range=0.0, 
            bias_init=-17.63, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        prng_key=head_prng_key, 
        pad_to_output_dim=None, 
        params_dtype=jnp.float32, 
    )

    loss_f = partial(ppo_loss_fn, cliprange_value=cliprange_value, cliprange=cliprange, value_loss_coef=value_loss_coef)

    ppo_inference = GPT2ILQLInference.load_inference(
        initial_policy_params=initial_policy_params, 
        policy_params=policy_train_state.params, 
        value_head_params=value_head_train_state.params, 
        initial_policy_model=policy_model, 
        policy_model=policy_model, 
        value_head_model=value_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_f, 
    )

    ppo_trainer = GPT2PPOTrain.load_train(
        policy_train_state=policy_train_state, 
        value_head_train_state=value_head_train_state, 
        policy_model=policy_model, 
        value_head_model=value_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_f, 
    )

    if use_adaptive_kl:
        kl_controller = AdaptiveKLController(init_kl_coef=init_kl_coef, target=kl_target, horizon=kl_horizon)
    else:
        kl_controller = FixedKLController(kl_coef=init_kl_coef)

    print("loading environment")
    prng_key, oracle_prng = jax.random.split(prng_key)
    env = TwentyQuestionsPolicyEnvironment(
        oracle=T5Oracle.load_oracle(
            mesh=mesh,
            prng_key=oracle_prng,
            model_load_mode=oracle_model_mode,
            model_load_path=oracle_model_path,
            use_fp16_activations=use_fp16_activations,
            use_fp16_params=use_fp16_params,
            max_input_length=124,
            max_output_length=4,
        ),
        word_list=get_default_word_list(),
        max_conversation_length=20,
    )

    if env_deterministic:
        def seed_generator():
            num_words = len(env.word_list)
            i = 0
            while True:
                yield i
                i = (i + 1) % num_words
    else:
        def seed_generator():
            random_state = random.Random(seed)
            while True:
                yield random_state.getrandbits(64)

    data_round = 0
    def ppo_dataset_loader(ppo_inference: GPT2ILQLInference, policy: GPT2Policy) -> PPODataset:
        nonlocal data_round

        interactions, summary_results = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=n_rollouts, 
            bsize=rollout_bsize, 
            seed_generator=seed_generator(),
            env_options={"deterministic": env_deterministic},
        )
        summary_results = pull_logs(summary_results)

        text_trajectories = []
        token_trajectory_chains = []
        for interaction in interactions:
            rewards = [0.0]
            for transition in interaction:
                rewards.append(transition.reward)
                rewards.append(0.0)

            text_history = interaction[-1].post_transition_history
            done = interaction[-1].done
            while True:
                text_trajectory = TextTrajectory(
                    text_history=text_history, 
                    reward=tuple(rewards),
                    done=done, 
                )
                token_trajectory = TokenTrajectory.from_text_trajectory(text_trajectory, tokenizer)
                if token_trajectory.tokens.shape[0] < max_input_length+max_output_length:
                    break

                # truncate one step
                text_history = text_history[:-2]
                last_r = rewards[-2]
                rewards = rewards[:-2]
                rewards[-2] += last_r * gamma
                done = False

            if token_trajectory.tokens.shape[0] == 0:
                continue
            text_trajectories.append(text_trajectory)
            token_trajectory_chains.append(TokenTrajectoryChain(token_trajectory, None))
        
        conversations = []
        for word_var, interaction in zip(env.word_list, interactions):
            final_text_history = interaction[-1].post_transition_history
            conversation = create_conversation_from_history(word_var, final_text_history, max_conversation_len=20)
            conversations.append(conversation)
        
        ppo_data, all_kls = ppo_inference.get_ppo_data_from_token_trajectory_chain(
            token_trajectory_chains, 
            bsize=ppo_data_bsize, 
            max_length=max_input_length+max_output_length, 
            gamma=gamma, 
            lam=lam, 
            kl_weight=kl_controller.value, 
            use_advantage_whitening=use_advantage_whitening, 
        )
        mean_kl = all_kls.mean().item()
        kl_controller.update(mean_kl, train_bsize)

        ppo_dataset = PPODataset.from_ppo_data_list(
            ppo_data, 
            tokenizer, 
            BlockingStrategy(Padding.RIGHT, Truncation.RIGHT, max_input_length+max_output_length), 
        )

        logs = dict(
            policy=dict(
                initial_policy_kl=get_tensor_stats_np(all_kls, np.ones(all_kls.shape), all_kls.size), 
                sqrt_initial_policy_kl=np.sqrt(mean_kl), 
                kl_ctrl_value=kl_controller.value, 
            ), 
            env_interaction=summary_results, 
        )

        logs = pull_logs(label_logs(logs, 'data_collection', {'round': data_round}))
        log(logs, use_wandb and is_main_process)

        if save_dir is not None and save_ppo_dataset:
            print('saving ppo dataset ...')
            data_save_path = os.path.join(save_dir, 'data_saves', f'{data_round}')
            if is_main_process:
                create_path(data_save_path)
            # save ppo_dataset
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'ppo_dataset.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(ppo_dataset, f)
            # save text_trajectory_chains
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'token_trajectory_chains.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(token_trajectory_chains, f)
            # save raw_results
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'interactions.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(interactions, f)
            # save conversations
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'conversations.json'), 
                enabled=is_main_process, 
            ), 'w') as f:
                json.dump(conversations, f, indent=2)
            # save summary_results
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'summary_results.json'), 
                enabled=is_main_process, 
            ), 'w') as f:
                json.dump(summary_results, f)
            print('done saving ppo dataset.')
        
        data_round += 1

        return ppo_dataset

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )
    
    prng_key, train_prng = jax.random.split(prng_key)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    ppo_trainer, ppo_inference, policy = train_loop(
        trainer=ppo_trainer, 
        inference=ppo_inference, 
        policy=policy, 
        load_dataset=ppo_dataset_loader, 
        evaluator=None, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        n_rounds=n_rounds, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_every_rounds=eval_every_rounds, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_every_rounds=save_every_rounds, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
