import jax.numpy as jnp
import optax
import jax
from flax.training.train_state import TrainState
from JaxSeq.utils import match_partition_rules
from JaxSeq.checkpointing import load_pytree
from functools import partial
from typing import Union, Callable
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jaxtyping import PyTree
from JaxSeq.utils import float_to_dtype
import flax.linen as nn

def shard_params_from_params(
    model: nn.Module, 
    params: PyTree, 
    mesh: Mesh, 
) -> PyTree:
    # get shard spec
    param_spec = match_partition_rules(model.config.get_partition_rules(), params)

    # get sharded params
    params = jax.tree_util.tree_map(lambda x, ps: jax.device_put(x, NamedSharding(mesh, ps)), params, param_spec)

    return params

def shard_train_state_from_params(
    model: nn.Module, 
    params: PyTree, 
    optim: optax.GradientTransformation, 
    mesh: Mesh, 
) -> TrainState:
    # setup train_state init function
    init_fn = lambda params: partial(TrainState.create, tx=optim, apply_fn=None)(params=params)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, params=params)
    train_state_spec = match_partition_rules(model.config.get_partition_rules(), train_state_shape)

    # get sharded train_state
    out_shardings = jax.tree_util.tree_map(lambda x: NamedSharding(mesh, x), train_state_spec)
    train_state = jax.jit(
        init_fn, 
        in_shardings=(out_shardings.params,), 
        out_shardings=out_shardings, 
        donate_argnums=(0,), 
    )(params)

    return train_state

def shard_params_from_config(
    model: nn.Module, 
    mesh: Mesh, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
) -> PyTree:
    # setup init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> PyTree:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return params

    # get shard spec
    params_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    param_spec = match_partition_rules(model.config.get_partition_rules(), params_shape)

    # get sharded params
    out_shardings = jax.tree_util.tree_map(lambda x: NamedSharding(mesh, x), param_spec)
    params = jax.jit(
        init_fn, 
        out_shardings=out_shardings, 
    )(jax.random.PRNGKey(0))

    return params

def shard_train_state_from_config(
    model: nn.Module, 
    optim: optax.GradientTransformation, 
    mesh: Mesh, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
) -> TrainState:
    
    # setup train_state init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> TrainState:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return TrainState.create(params=params, tx=optim, apply_fn=None)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    train_state_spec = match_partition_rules(model.config.get_partition_rules(), train_state_shape)

    # get sharded train_state
    out_shardings = jax.tree_util.tree_map(lambda x: NamedSharding(mesh, x), train_state_spec)
    train_state = jax.jit(
        init_fn, 
        out_shardings=out_shardings, 
    )(jax.random.PRNGKey(0))

    return train_state

def shard_params_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    mesh: Mesh, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
    stream_sharding: bool=True, # shard tensors as they are loaded
) -> PyTree:
    # setup init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> PyTree:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return params

    # get shard spec
    params_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    param_spec = match_partition_rules(model.config.get_partition_rules(), params_shape)

    # load params with sharding
    sharding = jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), param_spec)
    with jax.default_device(jax.devices('cpu')[0]):
        params = load_pytree(
            checkpoint_path, 
            target=params_shape, 
            dtype=params_dtype, 
            sharding=sharding if stream_sharding else None, 
        )

    if not stream_sharding:
        params = jax.tree_util.tree_map(lambda x, ps: jax.device_put(x, NamedSharding(mesh, ps)), params, param_spec)
    return params

def shard_train_state_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], # gets optim from params
    mesh: Mesh, 
    just_params: bool = False, 
    train_state_dtype: Union[str, jnp.dtype]=jnp.float32, 
    stream_sharding: bool=True, # shard tensors as they are loaded
) -> Union[TrainState, PyTree]:
    # setup train_state init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> TrainState:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        optim = optim_getter(params)
        return TrainState.create(params=params, tx=optim, apply_fn=None)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    train_state_spec = match_partition_rules(model.config.get_partition_rules(), train_state_shape)

    # load train_state
    sharding = jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), train_state_spec)
    with jax.default_device(jax.devices('cpu')[0]):
        train_state = load_pytree(
            checkpoint_path, 
            target=train_state_shape, 
            dtype=train_state_dtype, 
            sharding=sharding if stream_sharding else None, 
        )
    
    # get sharded params
    if just_params:
        if not stream_sharding:
            params = jax.tree_util.tree_map(lambda x, ps: jax.device_put(x, NamedSharding(mesh, ps)), train_state.params, train_state_spec.params)
        return params

    # get sharded train_state
    if not stream_sharding:
        train_state = jax.tree_util.tree_map(lambda x, ps: jax.device_put(x, NamedSharding(mesh, ps)), train_state, train_state_spec)
    return train_state

def shard_train_state_from_train_state(
    model: nn.Module,     
    train_state: TrainState, 
    mesh: Mesh, 
) -> TrainState:
    # get shard spec
    train_state_spec = match_partition_rules(model.config.get_partition_rules(), train_state)

    # get sharded train_state
    train_state = jax.tree_util.tree_map(lambda x, ps: jax.device_put(x, NamedSharding(mesh, ps)), train_state, train_state_spec)

    return train_state
