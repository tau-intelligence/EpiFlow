import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)

class Relu_StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        value = nn.softplus(value)

        return jnp.squeeze(value, -1)


class StateActionValueWithZ(nn.Module):
    """Epigraph action value Q_hat(x, z, a)."""
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, z: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, z, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)
