import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Optional, Union, Dict, Any
import re
from jax.sharding import PartitionSpec as PS

class Config:
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

# basic 2 layer MLP head

class MLPValueHeadConfig(Config):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        use_bias: bool=True, 
        initializer_range: Optional[int]=None, 
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        super().__init__()
    
    @staticmethod
    def get_partition_rules():
        return [
            (re.escape("['dense1']['kernel']"), PS(None, "mp")), 
            (re.escape("['dense1']['bias']"), PS("mp")), 
            (re.escape("['dense2']['kernel']"), PS("mp", None)), 
            (re.escape("['dense2']['bias']"), PS()), 
        ]

class MLPValueHead(nn.Module):
    config: MLPValueHeadConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        if self.config.initializer_range is None:
            self.initializer = jax.nn.initializers.lecun_normal()
        else:
            initalizer = jax.nn.initializers.normal(self.config.initializer_range)
        
        self.dense1 = nn.Dense(
            features=self.config.hidden_dim, 
            use_bias=self.config.use_bias, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            precision=self.precision, 
            kernel_init=initalizer, 
        )
        self.dense2 = nn.Dense(
            features=self.config.output_dim, 
            use_bias=self.config.use_bias, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            precision=self.precision, 
            kernel_init=initalizer, 
        )

    def __call__(
        self, 
        x: jax.Array, 
        *, 
        train: bool, 
    ) -> jax.Array:
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x

# basic linear head

class LinearValueHeadConfig(Config):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        use_bias: bool=True, 
        initializer_range: Optional[int]=None, 
        precision: Optional[Union[jax.lax.Precision, str]]=None, 

    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.precision = precision
        super().__init__()
    
    @staticmethod
    def get_partition_rules():
        return [
            (re.escape("['dense']['kernel']"), PS()), 
            (re.escape("['dense']['bias']"), PS()), 
        ]

class LinearValueHead(nn.Module):
    config: LinearValueHeadConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        if self.config.initializer_range is None:
            self.initializer = jax.nn.initializers.lecun_normal()
        else:
            initalizer = jax.nn.initializers.normal(self.config.initializer_range)
        
        self.dense = nn.Dense(
            features=self.config.output_dim, 
            use_bias=self.config.use_bias, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            precision=self.precision, 
            kernel_init=initalizer, 
        )

    def __call__(
        self, 
        x: jax.Array, 
        *, 
        train: bool, 
    ) -> jax.Array:
        x = self.dense(x)
        return x
