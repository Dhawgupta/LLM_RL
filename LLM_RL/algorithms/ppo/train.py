from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, Hashable, Iterator
from jaxtyping import PyTree
from jax.random import KeyArray
from collections import deque
import jax
from tqdm.auto import tqdm
from JaxSeq.utils import Dataset, dataloader, get_enabled_save_path, create_path
from LLM_RL.algorithms.ppo.data import PPODataset, PPOIterableDataset
from LLM_RL.algorithms.ppo.base_interface import PPOTrain, PPOInference
from JaxSeq.logs import combine_logs, label_logs, log, pull_logs
from JaxSeq.shard_model import get_sharding_from_model as get_sharding_from_model_policy
from LLM_RL.heads.shard_heads import get_sharding_from_model as get_sharding_from_model_head
import os
import wandb
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.bucket_manager import delete_with_bucket as delete
from JaxSeq.checkpointing import save_pytree
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import pickle as pkl
from LLM_RL.environment import TextEnv
from LLM_RL.algorithms.ppo.base_interface import PPOPolicy
import flax.linen as nn
import os

def dump_state(
    policy_model: FlaxPreTrainedModel, 
    policy_train_state: TrainState, 
    value_head_model: nn.Module, 
    value_head_train_state: TrainState, 
    save_dir: str, 
    save_train_state: bool, 
    enable_save: bool, 
    **loop_state: Dict[Hashable, Any], 
):
    # dump loop_state
    with open(get_enabled_save_path(os.path.join(save_dir, 'loop_state.pkl'), enabled=enable_save), 'wb') as f:
        pkl.dump(loop_state, f)
    
    # save policy
    if enable_save:
        create_path(os.path.join(save_dir, 'policy'))
    # dump policy config
    with open(get_enabled_save_path(os.path.join(save_dir, 'policy', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(policy_model.config.to_json_string())
    # dump policy_train_state
    if save_train_state:
        save_pytree(
            tree=policy_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'policy', 'train_state.msgpack'), enabled=enable_save), 
            sharding=get_sharding_from_model_policy(policy_model, policy_train_state), 
        )
    else:
        save_pytree(
            tree=policy_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'policy', 'params.msgpack'), enabled=enable_save), 
            sharding=get_sharding_from_model_policy(policy_model, policy_train_state.params), 
        )

    # save value head
    if enable_save:
        create_path(os.path.join(save_dir, 'value_head'))
    # dump value_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'value_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(value_head_model.config.to_json_string())
    # dump value_head_train_state
    if save_train_state:
        save_pytree(
            tree=value_head_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'value_head', 'train_state.msgpack'), enabled=enable_save), 
            sharding=get_sharding_from_model_head(value_head_model, value_head_train_state), 
        )
    else:
        save_pytree(
            tree=value_head_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'value_head', 'params.msgpack'), enabled=enable_save), 
            sharding=get_sharding_from_model_head(value_head_model, value_head_train_state.params), 
        )

def eval_loss(
    inference: PPOInference, 
    dataset: Union[PPODataset, PPOIterableDataset], 
    prng_key: Optional[KeyArray], 
    bsize: int, 
    eval_batches: Optional[int], 
) -> Dict[str, Any]:
    # setup evaluator loop state
    eval_logs = []

    # eval on batches
    prng_key, new_prng = jax.random.split(prng_key) if prng_key is not None else (None, None)
    d = dataloader(new_prng, dataset, bsize, truncate=True)
    for i, batch in tqdm(enumerate(d)):
        # conditionally terminate early
        if eval_batches is not None and i >= eval_batches:
            break

        # get eval logs
        _, info = inference.eval_loss(**batch)
        eval_logs.append(info)
    
    # gather and postproc eval logs
    eval_logs = pull_logs(combine_logs(eval_logs))
    return eval_logs

def train_loop(
    trainer: PPOTrain, 
    inference: PPOInference, 
    policy: PPOPolicy, 
    load_dataset: Callable[[PPOInference, PPOPolicy], Union[PPODataset, PPOIterableDataset]], 
    evaluator: Optional[Callable[[PPOInference, PPOPolicy], Tuple[float, Dict[str, Any]]]], 
    prng_key: KeyArray, 
    save_dir: Optional[str], 
    n_rounds: int, 
    epochs: int, 
    max_steps: Optional[int], 
    bsize: int, 
    log_every: int, 
    eval_every_steps: Optional[int], 
    eval_every_epochs: Optional[int], 
    eval_every_rounds: Optional[int], 
    save_every_steps: Optional[int], 
    save_every_epochs: Optional[int], 
    save_every_rounds: Optional[int], 
    save_at_end: bool, 
    save_best: bool, 
    max_checkpoints: Optional[int], 
    save_train_state: bool, 
    use_wandb: bool, 
    wandb_project: Optional[str], 
    wandb_run_name: Optional[str], 
    wandb_config: Optional[Dict[str, Any]], 
    is_main_process: Optional[bool]=None, 
    **loop_state: Dict[Hashable, Any], 
) -> Tuple[PPOTrain, PPOInference, PPOPolicy]:
    assert (not use_wandb) or (use_wandb and wandb_project is not None)
    if is_main_process is None:
        is_main_process = is_main_process
    
    # initalize wandb
    wandb_id = loop_state.get('wandb_id', None)
    if use_wandb and is_main_process:
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        wandb.init(
            project=wandb_project, 
            id=wandb_id, 
            name=wandb_run_name, 
            config=wandb_config, 
            reinit=True, 
            resume="allow", 
        )

    # initalize training loop state
    train_logs = []
    best_perf = loop_state.get('best_perf', float('inf'))
    saved_checkpoints = loop_state.get('saved_checkpoints', deque([]))
    step = 0

    def _save(
        name: str, 
        add_to_queue: bool, 
        **loop_state: Dict[Hashable, Any], 
    ):
        nonlocal saved_checkpoints
        print(f'saving checkpoint {name} ...')
        # conditionally delete old checkpoints
        if add_to_queue and is_main_process:
            if (max_checkpoints is not None) and (len(saved_checkpoints) >= max_checkpoints):
                delete(saved_checkpoints.popleft(), recursive=True)
        curr_save_dir = os.path.join(save_dir, name)
        if is_main_process:
            create_path(curr_save_dir)
        dump_state(
            policy_model=trainer.policy_model, 
            policy_train_state=trainer.policy_train_state, 
            value_head_model=trainer.value_head_model, 
            value_head_train_state=trainer.value_head_train_state, 
            save_dir=curr_save_dir, 
            save_train_state=save_train_state, 
            enable_save=is_main_process, 
            **loop_state, 
        )
        if add_to_queue and is_main_process:
            saved_checkpoints.append(curr_save_dir)
        print('saved.')
    
    def _eval(
        **loop_state: Dict[Hashable, Any], 
    ):
        nonlocal best_perf
        nonlocal inference
        nonlocal policy
        # get eval logs
        inference = inference.replace(
            policy_params=trainer.policy_train_state.params, 
            value_head_params=trainer.value_head_train_state.params, 
        )
        policy.set_params(trainer.policy_train_state.params)
        eval_perf, eval_logs = evaluator(inference, policy)

        # publish eval logs
        eval_logs = pull_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch, 'round': round}))
        log(eval_logs, use_wandb and is_main_process)

        # conditionally save best model and optimizer state
        if save_dir is not None and save_best and eval_perf < best_perf:
            print('new best model!')
            best_perf = eval_perf
            _save(
                name='best', 
                add_to_queue=False, 
                **{**loop_state, 'best_perf': best_perf}, 
            )

    # begin training loop
    for round in tqdm(range(n_rounds)):

        # load dataset
        dataset = load_dataset(inference, policy)

        steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None
        if 'steps_per_epoch' in loop_state:
            assert steps_per_epoch == loop_state['steps_per_epoch'], 'loop_state steps_per_epoch does not match dataset steps_per_epoch'

        for epoch in tqdm(range(epochs)):
            prng_key, new_prng = jax.random.split(prng_key)
            d = dataloader(new_prng, dataset, bsize, truncate=True)
            for batch in tqdm(d, total=steps_per_epoch):
                
                # step model and get training logs
                prng_key, new_prng = jax.random.split(prng_key)
                if 'step' in loop_state and step < loop_state['step']:
                    step += 1
                    continue
                trainer, _, info = trainer.step(
                    **batch, 
                    prng_key=new_prng, 
                    train=True, 
                )
                train_logs.append(info)
                
                # publish training logs and clear logs
                if (step + 1) % log_every == 0:
                    logs = combine_logs(train_logs)
                    logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch, 'round': round}))
                    log(logs, use_wandb and is_main_process)
                    train_logs = []
                
                # begin evaluation
                if evaluator is not None and eval_every_steps is not None and (step + 1) % eval_every_steps == 0:
                    _eval(
                        # loop state metadata
                        best_perf=best_perf, 
                        step=step+1, 
                        epoch=epoch, 
                        round=round, 
                        saved_checkpoints=saved_checkpoints, 
                        steps_per_epoch=steps_per_epoch, 
                        wandb_id=wandb_id, 
                    )
                
                # periodically save checkpoint
                if save_dir is not None and save_every_steps is not None and (step + 1) % save_every_steps == 0:
                    _save(
                        name='step_%d' % (step+1), 
                        add_to_queue=True, 
                        # loop state metadata
                        best_perf=best_perf, 
                        step=step+1, 
                        epoch=epoch, 
                        round=round, 
                        saved_checkpoints=saved_checkpoints, 
                        steps_per_epoch=steps_per_epoch, 
                        wandb_id=wandb_id, 
                    )
                
                step += 1
                
                # conditionally terminate
                if max_steps is not None and step >= max_steps:
                    break
        
            # begin evaluation
            if evaluator is not None and eval_every_epochs is not None and (epoch + 1) % eval_every_epochs == 0:
                _eval(
                    # loop state metadata
                    best_perf=best_perf, 
                    step=step, 
                    epoch=epoch, 
                    round=round, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )
            
            # periodically save checkpoint
            if save_dir is not None and save_every_epochs is not None and (epoch + 1) % save_every_epochs == 0:
                _save(
                    name=f'epoch_{epoch}', 
                    add_to_queue=True, 
                    # loop state metadata
                    best_perf=best_perf, 
                    step=step, 
                    epoch=epoch, 
                    round=round, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )
            
            # conditionally terminate
            if max_steps is not None and step >= max_steps:
                break
        
        # begin evaluation
        if evaluator is not None and eval_every_rounds is not None and (round + 1) % eval_every_rounds == 0:
            _eval(
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                round=round, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        # periodically save checkpoint
        if save_dir is not None and save_every_rounds is not None and (round + 1) % save_every_rounds == 0:
            _save(
                name='round_%d' % (round), 
                add_to_queue=True, 
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                round=round, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        inference = inference.replace(
            policy_params=trainer.policy_train_state.params, 
            value_head_params=trainer.value_head_train_state.params, 
        )
        policy.set_params(trainer.policy_train_state.params)
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        _save(
            name='last', 
            add_to_queue=False, 
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            round=round, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )

    # stop wandb
    if use_wandb and is_main_process:
        wandb.finish()
    
    inference = inference.replace(
        policy_params=trainer.policy_train_state.params, 
        value_head_params=trainer.value_head_train_state.params, 
    )
    policy.set_params(trainer.policy_train_state.params)
    return trainer, inference, policy
