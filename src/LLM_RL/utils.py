from typing import Optional
import os
import jax.numpy as jnp
import jax
import numpy as np
from JaxSeq.utils import convert_path
from functools import partial

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
convert_path = partial(convert_path, project_root=PROJECT_ROOT)


def get_tensor_stats(xs: jax.Array, mask: jax.Array, n: int):
    """get stats about a tensor, used for logging"""
    mean = (xs * mask).sum() / n
    mask = mask.astype(jnp.bool_)
    return dict(
        mean=mean, 
        min=jnp.min(xs, where=mask, initial=float('inf')), 
        max=jnp.max(xs, where=mask, initial=float('-inf')), 
        std=jnp.std(xs, where=mask), 
    )

def unpad_array(xs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pad_t = jnp.where(1-mask)[0]
    if len(pad_t) > 0:
        xs = xs[:pad_t[0]]
    return xs
