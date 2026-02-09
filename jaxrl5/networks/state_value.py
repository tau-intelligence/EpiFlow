import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)

        return jnp.squeeze(value, -1)
    
class Relu_StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)

        value = nn.softplus(value)

        return jnp.squeeze(value, -1)


class StateValueWithZ(nn.Module):
    """State Value function that takes observations and an additional input z."""
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, z: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        # Concatenate observations with z along the last axis
        inputs = jnp.concatenate([observations, z], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)

        return jnp.squeeze(value, -1)

