"""Flow Matching Networks and Samplers."""
from functools import partial
from typing import Callable, Optional, Sequence, Type
import flax.linen as nn
import jax.numpy as jnp
import jax


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        cond_encoder_cls: Conditional encoder class for time encoding.
        reverse_encoder_cls: Reverse encoder class for the main network.
        time_preprocess_cls: Time preprocessing class (e.g., FourierFeatures).
    """
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 times: jnp.ndarray,
                 training: bool = False):
        """Return the velocity vectors at the given states, actions, and times.

        Args:
            observations: Observations (state).
            actions: Actions at time t.
            times: Time values in [0, 1].
            training: Whether in training mode.

        Returns:
            Predicted velocity vectors.
        """
        times = times.reshape(actions.shape[0], -1)  # (B, 1) shape check
        t_ff = self.time_preprocess_cls()(times)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        
        # Concatenate action, observation, and time encoding
        reverse_input = jnp.concatenate([actions, observations, cond], axis=-1)
        
        return self.reverse_encoder_cls()(reverse_input, training=training)


@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'num_steps', 'clip_sampler', 'training'))
def flow_matching_sampler(
    actor_apply_fn,
    actor_params,
    num_steps,
    rng,
    act_dim,
    observations,
    clip_sampler=True,
    training=False
):
    """Sample actions using flow matching with Euler integration.
    
    Args:
        actor_apply_fn: Actor network apply function.
        actor_params: Actor network parameters.
        num_steps: Number of integration steps.
        rng: Random number generator key.
        act_dim: Action dimension.
        observations: Observation batch.
        clip_sampler: Whether to clip actions to [-1, 1].
        training: Whether in training mode.
    
    Returns:
        Tuple of (sampled actions, updated rng).
    """
    batch_size = observations.shape[0]
    
    # Start from random noise (x_0 ~ N(0, I))
    key, rng = jax.random.split(rng, 2)
    actions = jax.random.normal(key, (batch_size, act_dim))
    
    # Euler integration from t=0 to t=1
    dt = 1.0 / num_steps
    
    def integration_step(carry, i):
        current_actions, current_rng = carry
        t = i * dt
        
        # Create time tensor
        time_tensor = jnp.full((batch_size, 1), t)
        
        # Get velocity prediction
        velocity = actor_apply_fn(
            {"params": actor_params},
            observations,
            current_actions,
            time_tensor,
            training=training
        )
        
        # Euler step: x_{t+dt} = x_t + v_t * dt
        current_actions = current_actions + velocity * dt
        
        if clip_sampler:
            current_actions = jnp.clip(current_actions, -1, 1)
        
        return (current_actions, current_rng), ()
    
    # Perform integration with unroll for faster execution
    (final_actions, rng), () = jax.lax.scan(
        integration_step,
        (actions, rng),
        jnp.arange(num_steps),
        unroll=5  # Unroll for faster JIT compilation and execution
    )
    
    # Final clip
    final_actions = jnp.clip(final_actions, -1, 1)
    
    return final_actions, rng


@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'num_steps', 'num_samples', 'clip_sampler'))
def flow_matching_sampler_with_qvalue(
    actor_apply_fn,
    actor_params,
    critic_apply_fn,
    critic_params,
    num_steps,
    num_samples,
    rng,
    act_dim,
    observations,
    clip_sampler=True
):
    """Sample multiple actions using flow matching and select based on Q-value.
    
    This is similar to IFQL's rejection sampling approach.
    
    Args:
        actor_apply_fn: Actor network apply function.
        actor_params: Actor network parameters.
        critic_apply_fn: Critic network apply function.
        critic_params: Critic network parameters.
        num_steps: Number of integration steps.
        num_samples: Number of action samples to generate.
        rng: Random number generator key.
        act_dim: Action dimension.
        observations: Single observation (will be repeated).
        clip_sampler: Whether to clip actions to [-1, 1].
    
    Returns:
        Tuple of (best action, updated rng).
    """
    # Repeat observations for multiple samples
    batch_observations = jnp.repeat(
        jnp.expand_dims(observations, 0),
        num_samples,
        axis=0
    )
    
    # Sample multiple actions
    actions, rng = flow_matching_sampler(
        actor_apply_fn,
        actor_params,
        num_steps,
        rng,
        act_dim,
        batch_observations,
        clip_sampler=clip_sampler,
        training=False
    )
    
    # Evaluate Q-values for all samples
    q_values = critic_apply_fn(
        {"params": critic_params},
        batch_observations,
        actions
    )
    
    # Select action with highest Q-value
    # If using ensemble, take minimum across ensemble
    if q_values.ndim > 1 and q_values.shape[0] > 1:
        q_values = q_values.min(axis=0)
    
    best_idx = jnp.argmax(q_values)
    best_action = actions[best_idx]
    
    return best_action, rng
