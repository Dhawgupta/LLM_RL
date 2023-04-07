from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask
import os
import optax
from JaxSeq.models.gptj.interface import GPTJTrain, GPTJInference
from JaxSeq.models.gptj.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from LLM_RL.algorithms.ppo.train import train_loop
from LLM_RL.algorithms.ppo.base_interface import ppo_loss_fn
from JaxSeq.generation_eval import generate_language, compute_metrics
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import TextEnv, TextHistory, Text, interact_environment, text_env_eval, TextTrajectory, TextTrajectoryChain
from LLM_RL.algorithms.ppo.gptj.interface import GPTJPolicy, GPTJPPOInference, GPTJPPOTrain
from LLM_RL.heads.linear_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.linear_head import LinearHeadConfig
from flax.training.train_state import TrainState
from LLM_RL.algorithms.ppo.data import PPODataset, PPOIterableDataset
from functools import partial

class BitsTestEnv(TextEnv):
    def __init__(self, n: int):
        self.n = n

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        try:
            bits = list(map(int, text_history[-1].text.strip().split(' ')))
        except:
            bits = []
        return text_history, float(sum(bits) > (self.n // 2))*10.0, True

    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None) -> TextHistory:
        return (Text(text='<|endoftext|>', is_action=False),)

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    epochs: int=1, 
    max_steps: Optional[int]=None, 
    
    lr: float=1e-5, 
    weight_decay: float=0.0, 

    train_bsize: int=16, 
    grad_accum_steps: int=1, 

    gradient_checkpoint: bool=False, 

    max_input_length: int=512, 
    max_output_length: int=512, 

    log_every: int=256, 
    eval_every: int=256, 

    eval_loss_bsize: int=32, 
    eval_loss_batches: Optional[int]=None, 

    generation_bsize: int=32, 
    generation_batches: Optional[int]=None, 
    generation_do_sample: bool=True, 
    generation_num_beams: int=1, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 

    ppo_data_bsize: int=32, 
    gamma: float=0.99, 
    lam: float=0.95, 
    kl_weight: float=1.0, # probably should change these defaults?

):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh(data_mesh_shape, model_mesh_shape)
    print(f"Mesh: {mesh}")

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
                b2=0.999, 
                eps=1e-6, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )

    model_dtype = get_dtype(use_fp16=jax.default_backend() == 'tpu')
    policy_train_state, policy_model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=model_dtype, 
        optim_getter=policy_optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        force_pad_embeddings=force_pad_embeddings, 
        gradient_checkpoint=gradient_checkpoint, 
        params_dtype=jnp.float32, 
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)

    policy_inference = GPTJInference.load_inference(
        params=policy_train_state.params, 
        model=policy_model, 
        tokenizer=tokenizer, 
        mesh=mesh, 
    )

    env = BitsTestEnv(n=10)
    
    policy_prng = jax.random.PRNGKey(0)
    policy = GPTJPolicy(
        inference=policy_inference, 
        prng_key=policy_prng, 
        generation_config=GenerationConfig(
            do_sample=True, 
            num_beams=1, 
            temperature=1.0, 
            top_p=1.0, 
            eos_token_id=tokenizer.encode('\n')[0], 
            pad_token_id=tokenizer.pad_token_id, 
            max_length=max_input_length+max_output_length, 
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
                b2=0.999, 
                eps=1e-6, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
    
    model_dtype = get_dtype(use_fp16=jax.default_backend() == 'tpu')
    value_head_train_state, value_head = load_head_train_state_from_config(
        model_config=LinearHeadConfig(
            input_dim=policy_model.config.n_embd, 
            output_dim=1, 
            use_bias=True, 
            initializer_range=None, 
        ), 
        model_dtype=model_dtype, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        pad_to_output_dim=None, 
        params_dtype=jnp.float32, 
    )

    loss_f = partial(ppo_loss_fn, cliprange_value=0.2, cliprange=0.2, value_loss_weight=1.0)

    ppo_inference = GPTJPPOInference.load_inference(
        initial_policy_params=policy_train_state.params, # possible that we have to copy this due to donation
        policy_params=policy_train_state.params, 
        value_head_params=value_head_train_state.params, 
        initial_policy_model=policy_model, 
        policy_model=policy_model, 
        value_head_model=value_head, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        loss_fn=loss_f, 
    )

    ppo_trainer = GPTJPPOTrain.load_train(
        policy_train_state=policy_train_state, 
        value_head_train_state=value_head_train_state, 
        policy_model=policy_model, 
        value_head_model=value_head, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        loss_fn=loss_f, 
    )

    raw_results, summary_results = text_env_eval(env=env, policy=policy, n_rounds=10)

    text_trajectory_chains = []
    for raw_result in raw_results:
        text_trajectory = TextTrajectory(
            text_history=raw_result[-1].post_transition_history, 
            reward=[0.0, raw_result[-1].reward], 
            done=raw_result[-1].done, 
        )
        text_trajectory_chains.append(TextTrajectoryChain(text_trajectory, None))
    
    ppo_data, all_kls = ppo_inference.get_ppo_data_from_text_trajectory_chain(
        text_trajectory_chains, 
        bsize=ppo_data_bsize, 
        max_length=max_input_length+max_output_length, 
        gamma=gamma, 
        lam=lam, 
        kl_weight=kl_weight, 
    )

    ppo_dataset = PPODataset.from_ppo_data_list(
        ppo_data, 
        tokenizer, 
        BlockingStrategy(Padding.RIGHT, Truncation.RIGHT, max_input_length+max_output_length), 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=outputs_path, 
        input_args=input_args, 
    )

    # eval_prng = jax.random.PRNGKey(0)
    # def evaluator(inference: GPTJInference):
    #     nonlocal eval_prng

    #     eval_prng, new_prng = jax.random.split(eval_prng)
    #     loss_metrics = eval_loss(
    #         inference=inference, 
    #         dataset=eval_data, 
    #         prng_key=new_prng, 
    #         bsize=eval_loss_bsize, 
    #         eval_batches=eval_loss_batches, 
    #     )

    #     eval_prng, new_prng = jax.random.split(eval_prng)
    #     generation_prompts = [
    #         eval_json_data[i] for i in jax.random.permutation(
    #             new_prng, 
    #             jnp.arange(len(eval_json_data)), 
    #         ).tolist()
    #     ]
    #     eval_prng, new_prng = jax.random.split(eval_prng)
    #     generation_data = generate_language(
    #         inference=inference, 
    #         prompts=list(map(lambda x: x['in_text'], generation_prompts)), 
    #         references=list(map(lambda x: [x['out_text']], generation_prompts)), 
    #         prng_key=new_prng, 
    #         bsize=generation_bsize, 
    #         generation_batches=generation_batches, 
    #         blocking_strategy=BlockingStrategy(
    #             padding=Padding.LEFT, 
    #             truncation=Truncation.LEFT, 
    #             max_length=max_input_length
    #         ), 
    #         generation_config=GenerationConfig(
    #             max_length=max_input_length+max_output_length, 
    #             do_sample=generation_do_sample, 
    #             num_beams=generation_num_beams, 
    #             pad_token_id=tokenizer.pad_token_id, 
    #             eos_token_id=tokenizer.eos_token_id, 
    #             temperature=1.0, 
    #             top_k=None, 
    #             top_p=None, 
    #         ), 
    #     )
    #     reference_metrics = compute_metrics(generation_data)

    #     return loss_metrics['loss'], {'loss_metrics': loss_metrics, 'reference_metrics': reference_metrics}
    
    train_prng = jax.random.PRNGKey(1)
    ppo_trainer, ppo_inference, policy = train_loop(
        trainer=ppo_trainer, 
        inference=ppo_inference, 
        policy=policy, 
        # evaluator=evaluator, 
        evaluator=None, 
        dataset=ppo_dataset, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every=eval_every, 
        save_every=None, 
        save_at_end=False, 
        save_best=True, 
        max_checkpoints=None, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
