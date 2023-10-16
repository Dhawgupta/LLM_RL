import contextlib
from typing import Any, Dict, List, Optional
import jax
from algorithms.jax_agent import Inference
from jax_utils.jax_shard import OptimType, shard_optim_and_params, shard_params
from jax_models.gpt2 import load_gpt2_model
import numpy as np
from jax.experimental.maps import Mesh
import optax
import dcargs
from functools import partial
from text_env_eval import text_env_eval
from token_history import text_history_to_token_history, text_transition_to_token_transition
from utils.path import convert_path
import os
import pickle as pkl
import json
from algorithms.jax_bc.core import bc_loss, load_bc_inference, load_bc_trainer
import pickle as pkl
from algorithms.jax_bc.data import BCDataset, filter_generator, filter_items
from algorithms.jax_bc.basic_train_loop import train_loop, eval_loop
from environments.car_dealer.data import create_trajectories_from_conversations, Role
import tree
from transformers import AutoTokenizer

def main(
    exp_name: Optional[str], 
    model_name: str, 

    /,  # Mark the end of positional arguments.

    role: Role=Role.SELLER, 

    top_p: Optional[float]=None, 

    checkpoint_path: Optional[str]=None, 
    checkpoint_is_sharded: bool=True, 

    data_path: Optional[str]='data/car_dealer', 
    output_path: Optional[str]='outputs/car_dealer', 

    use_wandb: bool=False, 
    wandb_project: Optional[str]='car-dealer-bc', 

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    epochs: int=2, 
    max_steps: Optional[int]=None, 
    eval_batches: Optional[int]=None, 
    
    use_adafactor: bool=False,
    lr: float=1e-5,
    use_lr_schedule: bool=False,
    peak_lr: float=5e-5, 
    end_lr: float=6e-5, 
    weight_decay: float=0.0, 

    train_bsize: int=32, 
    grad_accum_steps: int=1, 

    gradient_checkpoint: bool=True, 

    max_sequence_length: int=1024, 

    log_every: Optional[int]=None, 
    num_logs_per_epoch: int=10,
    eval_every: Optional[int]=None,
    num_evals_per_epoch: int=5, 
    save_every: Optional[int]=None, 
    num_saves_per_epoch: int=1,
    save_best: bool=False,
    save_best_also: bool=False,
    save_last: bool=False,

    inference_bsize: int=32, 
    seed: int=0,

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 
):
    if use_adafactor:
        assert weight_decay == 0.0, 'no weight decay with adafactor'
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    input_args = locals().copy()
    print(input_args)

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    with open(convert_path(os.path.join(data_path, 'train.json')), 'r') as f:
        raw_train = json.load(f)
    with open(convert_path(os.path.join(data_path, 'eval.json')), 'r') as f:
        raw_eval = json.load(f)

    train_text_trajectories = []
    eval_text_trajectories = []
    for personality, convos in raw_train.items():
        train_text_trajectories.extend(create_trajectories_from_conversations(convos, role))
    for personality, convos in raw_eval.items():
        eval_text_trajectories.extend(create_trajectories_from_conversations(convos, role))

    print(f"Initial dataset sizes: train: {len(train_text_trajectories)}, eval: {len(eval_text_trajectories)}")

    if top_p is not None:
        train_text_trajectories = filter_items(lambda x: sum(x.reward), train_text_trajectories, take_top=top_p, threshold=None)
        eval_text_trajectories = filter_items(lambda x: sum(x.reward), eval_text_trajectories, take_top=top_p, threshold=None)

    train_text_histories = [trajectory.text_history for trajectory in train_text_trajectories]
    eval_text_histories = [trajectory.text_history  for trajectory in eval_text_trajectories]

    train_token_histories = [text_history_to_token_history(text_history, tokenizer) for text_history in train_text_histories]
    eval_token_histories = [text_history_to_token_history(text_history, tokenizer) for text_history in eval_text_histories]
    
    train_token_histories = [token_history for token_history in train_token_histories if token_history.tokens.shape[0] <= max_sequence_length]
    eval_token_histories = [token_history for token_history in eval_token_histories if token_history.tokens.shape[0] <= max_sequence_length]

    print(f"Final dataset sizes: train: {len(train_token_histories)}, eval: {len(eval_token_histories)}")

    train_data = BCDataset(
        token_histories=train_token_histories, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )

    eval_data = BCDataset(
        token_histories=eval_token_histories, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )
    
    if checkpoint_is_sharded and checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
        checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)
    
    if model_name == 'gpt2-xl' or model_name == 'gpt2-medium':
        print("loading model")
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=gradient_checkpoint, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    else:
        raise NotImplementedError

    n_datapoints = len(train_text_histories)
    n_steps = int(min((n_datapoints//(train_bsize*grad_accum_steps))*epochs, max_steps if max_steps is not None else float('inf')))

    if use_lr_schedule:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, 
            peak_value=peak_lr, 
            warmup_steps=(n_steps)//10, 
            decay_steps=n_steps-(n_steps//10), 
            end_value=end_lr, 
        )
    else:
        lr_schedule = lr
    
    if use_adafactor:
        optim = optax.MultiSteps(
            optax.adafactor(
                learning_rate=lr_schedule, 
                multiply_by_parameter_scale=False, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
        optim_type = OptimType.AdaFactorMultiStep
    else:
        optim = optax.MultiSteps(
            optax.adamw(
                learning_rate=lr_schedule, 
                b1=0.9, 
                b2=0.999, 
                eps=1e-8, 
                weight_decay=weight_decay, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
        optim_type = OptimType.AdamWMultiStep

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
    else:
        mesh = contextlib.nullcontext()

    # shard params and optimizer
    if do_pjit:
        (params, param_spec), (optim_state, optim_state_spec) = shard_optim_and_params(partial(model.init_weights, input_shape=(1, 1)), 
                                                                                       params, shard_rules, mesh, optim, optim_type)
    else:
        optim_state, param_spec, optim_state_spec = optim.init(params), None, None

    loss_fn = partial(bc_loss, non_action_weight=0.0)

    print("loading trainer and inference")    
    trainer = load_bc_trainer(
        model=model, 
        params=params, 
        param_spec=param_spec, 
        tokenizer=tokenizer, 
        optim=optim, 
        optim_state=optim_state, 
        optim_state_spec=optim_state_spec, 
        do_pjit=do_pjit, 
        loss_fn=loss_fn, 
    )

    inference = load_bc_inference(
        model=model, 
        params=params, 
        param_spec=param_spec, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=loss_fn, 
    )

    rng = jax.random.PRNGKey(seed)

    save_dir = None
    if exp_name is not None:
        save_dir = convert_path(os.path.join(output_path, exp_name, 'shard_%d' % (jax.process_index())))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        
        # copy training script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
            pkl.dump(input_args, f)
        
        # save info about mesh devices
        if do_pjit:
            with open(os.path.join(save_dir, 'system_mesh.pkl'), 'wb') as f:
                pkl.dump({'mesh': tree.map_structure(lambda x: {'id': int(x.id), 'process_index': int(x.process_index)}, mesh.devices.tolist()), 
                          'process_index': int(jax.process_index()), 'process_count': int(jax.process_count())}, f)
    
    rng, evaluator_rng = jax.random.split(rng)
    def evaluator(inference: Inference):
        nonlocal evaluator_rng
        
        evaluator_rng, eval_loop_rng = jax.random.split(evaluator_rng)
        data_results = eval_loop(
            inference=inference, 
            dataset=eval_data, 
            rng=eval_loop_rng, 
            bsize=inference_bsize, 
            prefetch_batches=None, 
            eval_batches=eval_batches, 
        )

        return data_results['loss'], {'data': data_results}

    if log_every is None:
        log_every = n_datapoints // (train_bsize * num_logs_per_epoch)
    if eval_every is None:
        eval_every = n_datapoints // (train_bsize * num_evals_per_epoch)
    if save_every is None:
        save_every = n_datapoints // (train_bsize * num_saves_per_epoch)
    
    if save_best and not save_last:
        save_every = None

    rng, training_rng = jax.random.split(rng)

    with mesh:
        trainer, inference = train_loop(
            model=model, 
            trainer=trainer, 
            inference=inference, 
            evaluator=evaluator, 
            dataset=train_data, 
            rng=training_rng, 
            save_dir=save_dir, 
            max_checkpoints=1 if save_last else None, 
            epochs=epochs, 
            max_steps=max_steps, 
            bsize=train_bsize, 
            prefetch_batches=None, 
            log_every=log_every, 
            eval_every=eval_every, 
            save_every=save_every, 
            save_at_end=save_last, 
            save_best=save_best or save_best_also, 
            use_wandb=use_wandb, 
            wandb_project=wandb_project, 
            wandb_run_name=exp_name, 
            wandb_config=None, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )

if __name__ == "__main__":
    dcargs.cli(main)