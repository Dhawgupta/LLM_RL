from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, Hashable, Iterator
from jaxtyping import PyTree
from jax.random import KeyArray
from collections import deque
import jax
from tqdm.auto import tqdm
from JaxSeq.utils import Dataset, dataloader
from LLM_RL.algorithms.ppo.data import PPODataset, PPOIterableDataset
from LLM_RL.algorithms.ppo.base_interface import PPOTrain, PPOInference
from JaxSeq.logs import combine_logs, label_logs, log, pull_logs
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

# def dump_state(
#     model: FlaxPreTrainedModel, 
#     train_state: TrainState, 
#     save_dir: str, 
#     enable_save: bool, 
#     **loop_state: Dict[Hashable, Any], 
# ):
#     # fetch train_state from devices/hosts
#     train_state = jax.device_get(train_state)
#     if enable_save:
#         # dump model config
#         with open(os.path.join(save_dir, 'config.json'), 'w') as f:
#             f.write(model.config.to_json_string())
#         # dump loop_state
#         with open(os.path.join(save_dir, 'loop_state.pkl'), 'wb') as f:
#             pkl.dump(loop_state, f)
#         # dump train_state
#         save_pytree(
#             tree=train_state, 
#             path=os.path.join(save_dir, 'train_state.msgpack'), 
#         )

# def eval_loss(
#     inference: PPOInference, 
#     dataset: Union[PPODataset, PPOIterableDataset], 
#     prng_key: Optional[KeyArray], 
#     bsize: int, 
#     eval_batches: Optional[int], 
# ) -> Dict[str, Any]:
#     # setup evaluator loop state
#     eval_logs = []

#     # eval on batches
#     prng_key, new_prng = jax.random.split(prng_key) if prng_key is not None else (None, None)
#     d = dataloader(new_prng, dataset, bsize, truncate=True)
#     for i, batch in tqdm(enumerate(d)):
#         # conditionally terminate early
#         if eval_batches is not None and i >= eval_batches:
#             break

#         # get eval logs
#         _, info = inference.eval_loss(**batch)
#         eval_logs.append(info)
    
#     # gather and postproc eval logs
#     eval_logs = pull_logs(combine_logs(eval_logs))
#     return eval_logs

def train_loop(
    ppo_trainer: PPOTrain, 
    ppo_inference: PPOInference, 
    policy: PPOPolicy, 
    env: TextEnv, 
    evaluator: Optional[Callable[[PPOInference, PPOPolicy], Tuple[float, Dict[str, Any]]]], 
    prng_key: KeyArray, 
    save_dir: Optional[str], 
    epochs: int, 
    max_steps: Optional[int], 
    bsize: int, 
    log_every: int, 
    eval_every: int, 
    save_every: Optional[int], 
    save_at_end: bool, 
    save_best: bool, 
    max_checkpoints: Optional[int], 
    use_wandb: bool, 
    wandb_project: Optional[str], 
    wandb_run_name: Optional[str], 
    wandb_config: Optional[Dict[str, Any]], 
    **loop_state: Dict[Hashable, Any], 
) -> Tuple[PPOTrain, PPOInference, PPOPolicy]:
    assert (not use_wandb) or (use_wandb and wandb_project is not None)
    
    # initalize wandb
    wandb_id = loop_state.get('wandb_id', None)
    if use_wandb and jax.process_index() == 0:
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
    steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None
    if 'steps_per_epoch' in loop_state:
        assert steps_per_epoch == loop_state['steps_per_epoch'], 'loop_state steps_per_epoch does not match dataset steps_per_epoch'

    # begin training loop
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
                logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                log(logs, use_wandb and jax.process_index() == 0)
                train_logs = []
            
            # begin evaluation
            if (evaluator is not None) and (step + 1) % eval_every == 0:

                # get eval logs
                inference = inference.replace(params=trainer.train_state.params)
                eval_perf, eval_logs = evaluator(inference)

                # publish eval logs
                eval_logs = pull_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch}))
                log(eval_logs, use_wandb and jax.process_index() == 0)

                # conditionally save best model and optimizer state
                if save_dir is not None and save_best and eval_perf < best_perf:
                    print('new best model! Saving ...')
                    curr_save_dir = os.path.join(save_dir, 'best')
                    if (not save_dir.startswith('gcs://')) and (not os.path.exists(curr_save_dir)):
                        os.makedirs(curr_save_dir)
                    dump_state(
                        model=trainer.model, 
                        train_state=trainer.train_state, 
                        save_dir=curr_save_dir, 
                        enable_save=jax.process_index() == 0, 
                        # loop state metadata
                        best_perf=eval_perf, 
                        step=step+1, 
                        epoch=epoch, 
                        saved_checkpoints=saved_checkpoints, 
                        steps_per_epoch=steps_per_epoch, 
                        wandb_id=wandb_id, 
                    )
                    print('saved.')
                    best_perf = eval_perf
            
            # periodically save checkpoint
            if save_dir is not None and save_every is not None and (step + 1) % save_every == 0:
                print('saving checkpoint...')
                # conditionally delete old checkpoints
                if (max_checkpoints is not None) and (len(saved_checkpoints) >= max_checkpoints):
                    delete(saved_checkpoints.popleft(), recursive=True)
                curr_save_dir = os.path.join(save_dir, 'step_%d' % (step+1))
                if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
                    os.makedirs(curr_save_dir)
                dump_state(
                    model=trainer.model, 
                    train_state=trainer.train_state, 
                    save_dir=curr_save_dir, 
                    enable_save=jax.process_index() == 0, 
                    # loop state metadata
                    best_perf=best_perf, 
                    step=step+1, 
                    epoch=epoch, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )
                saved_checkpoints.append(curr_save_dir)
                print('saved.')
            
            # conditionally terminate
            if max_steps is not None and (step + 1) >= max_steps:
                break

            step += 1
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        print('saving checkpoint...')
        curr_save_dir = os.path.join(save_dir, 'step_%d' % (step+1))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(curr_save_dir)
        dump_state(
            model=trainer.model, 
            train_state=trainer.train_state, 
            save_dir=curr_save_dir, 
            enable_save=jax.process_index() == 0, 
            # loop state metadata
            best_perf=best_perf, 
            step=step+1, 
            epoch=epoch, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )
        print('saved.')

    # stop wandb
    if use_wandb and jax.process_index() == 0:
        wandb.finish()
    
    inference = inference.replace(params=trainer.train_state.params)
    return trainer, inference
