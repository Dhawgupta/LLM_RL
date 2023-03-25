from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask
import os
import optax
from JaxSeq.models.gptj.interface import GPTJTrain, GPTJInference
from JaxSeq.models.gptj.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from JaxSeq.train import eval_loss, train_loop
from JaxSeq.generation_eval import generate_language, compute_metrics
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re

def setup_experiment_save(
    exp_name: Optional[str], 
    outputs_path: Optional[str], 
    input_args: Dict[str, Any], 
) -> Tuple[Optional[str], str]:
    save_dir = None
    if exp_name is None:
        exp_name = uuid_name(base_name="exp", include_uuid=False)
    if outputs_path is not None:
        save_dir = convert_path(os.path.join(outputs_path, exp_name))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        
        # copy training script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
            pkl.dump(input_args, f)
    return save_dir, exp_name

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 
    eval_data_path: str, 

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
):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh(data_mesh_shape, model_mesh_shape)
    print(f"Mesh: {mesh}")

    # load data
    # with open(convert_path(train_data_path), 'r') as f:
    #     train_json_data = jsonl_load(f)
    # with open(convert_path(eval_data_path), 'r') as f:
    #     eval_json_data = jsonl_load(f)

    # train_data = Seq2SeqDataset.from_str_list(
    #     list(map(lambda x: (x['in_text'], x['out_text']), train_json_data)), 
    #     tokenizer, 
    #     in_blocking_strategy=BlockingStrategy(
    #         padding=Padding.LEFT, 
    #         truncation=Truncation.LEFT, 
    #         max_length=max_input_length
    #     ), 
    #     out_blocking_strategy=BlockingStrategy(
    #         padding=Padding.RIGHT, 
    #         truncation=Truncation.RIGHT, 
    #         max_length=max_output_length
    #     ), 
    # )

    # eval_data = Seq2SeqDataset.from_str_list(
    #     list(map(lambda x: (x['in_text'], x['out_text']), eval_json_data)), 
    #     tokenizer, 
    #     in_blocking_strategy=BlockingStrategy(
    #         padding=Padding.LEFT, 
    #         truncation=Truncation.LEFT,
    #         max_length=max_input_length
    #     ), 
    #     out_blocking_strategy=BlockingStrategy(
    #         padding=Padding.RIGHT, 
    #         truncation=Truncation.RIGHT, 
    #         max_length=max_output_length
    #     ), 
    # )

    def optim_getter(params: PyTree):
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
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=model_dtype, 
        optim_getter=optim_getter, 
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
    
    trainer = GPTJTrain.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
        mesh=mesh, 
    )

    inference = GPTJInference.load_inference(
        params=train_state.params, 
        model=model, 
        tokenizer=tokenizer, 
        mesh=mesh, 
    )

    # save_dir, exp_name = setup_experiment_save(
    #     exp_name=exp_name, 
    #     outputs_path=outputs_path, 
    #     input_args=input_args, 
    # )

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
    
    # train_prng = jax.random.PRNGKey(1)
    # trainer, inference = train_loop(
    #     trainer=trainer, 
    #     inference=inference, 
    #     evaluator=evaluator, 
    #     dataset=train_data, 
    #     prng_key=train_prng, 
    #     save_dir=save_dir, 
    #     epochs=epochs, 
    #     max_steps=max_steps, 
    #     bsize=train_bsize, 
    #     log_every=log_every, 
    #     eval_every=eval_every, 
    #     save_every=None, 
    #     save_at_end=False, 
    #     save_best=True, 
    #     max_checkpoints=None, 
    #     use_wandb=use_wandb, 
    #     wandb_project=wandb_project, 
    #     wandb_run_name=exp_name, 
    #     wandb_config=None, 
    #     **loop_state, 
    # )

if __name__ == "__main__":
    tyro.cli(main)
