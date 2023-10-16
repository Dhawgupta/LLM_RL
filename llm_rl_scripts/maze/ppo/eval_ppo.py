from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import load_mesh, get_dtype
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import text_env_eval
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2ILQLPolicy, GPT2ILQLInference
from LLM_RL.heads.linear_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.linear_head import LinearHeadConfig
from JaxSeq.shard_model import shard_params_from_params
from JaxSeq.logs import pull_logs
import json
from JaxSeq.utils import multihost_device_get
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env, pick_start_position, compute_move_accuracy
from llm_rl_scripts.maze.env.mazes import double_t_maze
from IPython import embed


def main(
    exp_name: str="ppo_online_endgames_queen_rook",
    model_load_mode: ModelLoadMode=ModelLoadMode.TRAIN_STATE,
    model_load_path: str="outputs/chess/test_bc_shuffled2/model/",
    checkpoint_dir: str="/home/isadoracw/isadoracw/LLM_RL/outputs/chess/lr1e-6_ppo_online_endgames_queen_rook/lr1e-6_ppo_online_endgames_queen_rook.2023-06-05-07-13-26.269.76fee0fa037011eeb99cbd216c6583a9/round_99",
    
    # random_positions: bool=False,
    # save_positions:bool=False,
    output_path: str="logs/maze/",
    data_mesh_shape:int=1,
    fsdp_mesh_shape:int=1,
    model_mesh_shape:int=1,
    
    policy_do_sample:bool=True,
    policy_num_beams:int=1, 
    policy_temperature:Optional[float]=None,
    policy_top_p:Optional[float]=None,
    policy_top_k:Optional[int]=None,
    policy_max_new_tokens:int=512,
    max_output_length:int=512,
    max_eval_length:int=512,
    
    n_rollouts:int=16,
    full_games:bool=False,
    random_opponent:bool=False,
    
    maze_name:str="umaze",
    describe_function:str="describe_observation",
    
):
    # get checkpoint directory
    # checkpoint_dir = "~/isadoracw/LLM_RL/outputs/chess/ppo_online_endgames_queen_rook/ppo_online_endgames_queen_rook.2023-06-04-22-46-25.986.a317fd12032911ee9fce87d7217c0314/round_59/"
    # policy_path = os.path.join(checkpoint_dir, "policy", "train_state.msgpack")
    # value_head_path = os.path.join(checkpoint_dir, "value_head", "train_state.msgpack")

    # # load checkpoints from checkpoint directory
    # target = TrainState 

    # policy_train_state = load_pytree(policy_path, target=target)
    # policy_params = policy_train_state["params"]

    # value_head_train_state = load_pytree(value_head_path, target=target)
    # value_head_params = value_head_train_state["params"]

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
                learning_rate=1e-5, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=0.0, 
                mask=mask, 
            ), 
            every_k_schedule=4, 
        )
    
    model_dtype = get_dtype(use_fp16=False)
    params_dtype = get_dtype(use_fp16=False)
    
    model_prng_key = jax.random.PRNGKey(2)

    model_load_path = os.path.join(checkpoint_dir, "policy")
    embed()
    policy_train_state, policy_model = load_train_state(
            model_load_mode=model_load_mode, 
            model_load_path=model_load_path,
            model_dtype=model_dtype, 
            optim_getter=policy_optim_getter, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=model_prng_key, 
            force_pad_embeddings=False, 
            params_dtype=params_dtype, 
        )
    
    with jax.default_device(jax.devices('cpu')[0]):
        initial_policy_params = jax.tree_util.tree_map(
            lambda x: multihost_device_get(x, mesh=mesh).copy(), 
            policy_train_state.params, 
        )
    initial_policy_params = shard_params_from_params(
        model=policy_model, 
        params=initial_policy_params, 
    )

    
    policy_inference = GPT2Inference.load_inference(
        params=policy_train_state.params, 
        model=policy_model, 
        tokenizer=tokenizer, 
    )
    
    policy_prng = jax.random.PRNGKey(0)
    policy = GPT2ILQLPolicy(
        inference=policy_inference, 
        prng_key=policy_prng, 
        generation_config=GenerationConfig(
            do_sample=policy_do_sample, 
            num_beams=1, 
            temperature=policy_temperature, 
            top_p=policy_top_p, 
            top_k=policy_top_k, 
            eos_token_id=tokenizer.encode('\n')[0], 
            pad_token_id=tokenizer.pad_token_id, 
            max_new_tokens=policy_max_new_tokens, 
        ), 
        blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_output_length, 
        ), 
        out_str_process=lambda x: x.removesuffix('\n')+'\n', 
    )
    
    def value_head_optim_getter(params: PyTree):
        mask = get_weight_decay_mask(("bias",))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=1e-5, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=0.0, 
                mask=mask, 
            ), 
            every_k_schedule=4, 
        )
    
    head_prng_key = jax.random.PRNGKey(3)
    value_head_train_state, value_head = load_head_train_state_from_config(
        model_config=LinearHeadConfig(
            input_dim=policy_model.config.n_embd, 
            output_dim=1, 
            use_bias=True, 
            initializer_range=0.0, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        prng_key=head_prng_key, 
        pad_to_output_dim=None, 
        params_dtype=jnp.float32, 
    )
    
    ppo_inference = GPT2ILQLInference.load_inference(
        initial_policy_params=initial_policy_params, 
        policy_params=policy_train_state.params, 
        value_head_params=value_head_train_state.params, 
        initial_policy_model=policy_model, 
        policy_model=policy_model, 
        value_head_model=value_head, 
        tokenizer=tokenizer, 
        loss_fn=None, 
    )
    
    env = setup_maze_env(maze_name=maze_name, describe_function=describe_function, reward_function="standard_reward", last_k=1)
    start_position = pick_start_position(maze_name=maze_name)
    
    maze = double_t_maze()
    with mesh:
        accuracy = compute_move_accuracy(maze, policy, reranker=False)
    
    
    raw_results, summary_results = text_env_eval(
            env=env,
            policy=policy,
            n_rollouts=16,
            bsize=16,
            env_options={"init_position": start_position},
        )
    # check output directory
    save_dir = None
    if output_path is not None:
        # save_dir = convert_path(os.path.join(output_path, exp_name))
        save_dir = os.path.join(os.getcwd(), output_path, exp_name)
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            print(f"Output directory {save_dir} does not exist. Making directory...")
            os.makedirs(save_dir)
        
        # copy script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        # with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
        #     pkl.dump(input_args, f)
    
    summary_results = pull_logs(summary_results)
    print(summary_results)
    summary_results["accuracy"] = accuracy
    
    with open(os.path.join(output_path, exp_name, "results.jsonl"), "w") as f:
        f.write(json.dumps(summary_results) + "\n")
    
if __name__ == "__main__":
    tyro.cli(main) 

    

    #  