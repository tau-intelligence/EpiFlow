"""Implementations of algorithms for continuous control using Flow Matching."""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateActionValueWithZ, StateValue, StateValueWithZ, Relu_StateActionValue, Relu_StateValue, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule, ActorVectorField, flow_matching_sampler
from jaxrl5.networks.diffusion import dpm_solver_sampler_1st, vp_sde_schedule


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class DiffCBF(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    qz_critic: TrainState
    target_qz_critic: TrainState
    vz: TrainState
    qz1_critic: TrainState
    target_qz1_critic: TrainState
    vz1: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        sampling_method: str = 'flow_matching',  # Changed from 'ddpm' to 'flow_matching'
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key, qz_critic_key, vz_key, qz1_critic_key, vz1_key = jax.random.split(rng, 10)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=actor_layer_norm,
                                    activate_final=False)
            
            # Use flow matching instead of DDPM
            actor_def = ActorVectorField(time_preprocess_cls=preprocess_time_cls,
                                        cond_encoder_cls=cond_model_cls,
                                        reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=actor_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=actor_dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)
            
            # Use flow matching instead of DDPM
            actor_def = ActorVectorField(time_preprocess_cls=preprocess_time_cls,
                                        cond_encoder_cls=cond_model_cls,
                                        reverse_encoder_cls=base_model_cls)

        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx = optax.adamw(
                                            learning_rate=actor_lr, 
                                            weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                        ))
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)

            # critic_cls = partial(Relu_StateActionValue, base_cls=critic_base_cls)
            # critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )


        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)
            # value_def = Relu_StateValue(base_cls=value_base_cls)

        safe_value_params = value_def.init(safe_value_key, observations)["params"]

        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        # Initialize qz_critic (StateActionValueWithZ) - takes (obs, z, action) as inputs
        # z is sampled from [-49.3, 0], so z_dim = 1
        dummy_z = jnp.zeros((1, 1))  # Shape (batch_size=1, z_dim=1)
        qz_critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        qz_critic_cls = partial(StateActionValueWithZ, base_cls=qz_critic_base_cls)
        qz_critic_def = Ensemble(qz_critic_cls, num=num_qs)
        qz_critic_params = qz_critic_def.init(qz_critic_key, observations, dummy_z, actions)["params"]
        qz_critic_optimiser = optax.adam(learning_rate=critic_lr)
        qz_critic = TrainState.create(
            apply_fn=qz_critic_def.apply,
            params=qz_critic_params,
            tx=qz_critic_optimiser,
        )
        target_qz_critic = TrainState.create(
            apply_fn=qz_critic_def.apply,
            params=qz_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Initialize vz (StateValueWithZ) - takes (obs, z) as inputs
        vz_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        vz_def = StateValueWithZ(base_cls=vz_base_cls)
        vz_params = vz_def.init(vz_key, observations, dummy_z)["params"]
        vz = TrainState.create(apply_fn=vz_def.apply,
                               params=vz_params,
                               tx=value_optimiser)

        # Initialize qz1_critic (StateActionValueWithZ) - learns min(V - z, -Vc)
        qz1_critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        qz1_critic_cls = partial(StateActionValueWithZ, base_cls=qz1_critic_base_cls)
        qz1_critic_def = Ensemble(qz1_critic_cls, num=num_qs)
        qz1_critic_params = qz1_critic_def.init(qz1_critic_key, observations, dummy_z, actions)["params"]
        qz1_critic = TrainState.create(
            apply_fn=qz1_critic_def.apply,
            params=qz1_critic_params,
            tx=qz_critic_optimiser,
        )
        target_qz1_critic = TrainState.create(
            apply_fn=qz1_critic_def.apply,
            params=qz1_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Initialize vz1 (StateValueWithZ) - learns min(V - z, -Vc)
        vz1_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        vz1_def = StateValueWithZ(base_cls=vz1_base_cls)
        vz1_params = vz1_def.init(vz1_key, observations, dummy_z)["params"]
        vz1 = TrainState.create(apply_fn=vz1_def.apply,
                                params=vz1_params,
                                tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            qz_critic=qz_critic,
            target_qz_critic=target_qz_critic,
            vz=vz,
            qz1_critic=qz1_critic,
            target_qz1_critic=target_qz1_critic,
            vz1=vz1,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            sampling_method=sampling_method,
            critic_objective=critic_objective,
            critic_type=critic_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])

            if agent.critic_objective == 'expectile':
                value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f'Invalid critic objective: {agent.critic_objective}')

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)

        agent = agent.replace(value=value)

        return agent, info
    
    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = 0.1 * batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info
    
    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()

            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        agent = agent.replace(safe_value=safe_value)

        return agent, info
    
    def update_vz(agent, batch: DatasetDict, rng: jax.random.PRNGKey) -> Tuple[Agent, Dict[str, float]]:
        """Update vz (value function with z) using qz_critic. z is sampled from [-49.3, 0]."""
        # Sample z_t from [-49.3, 0]
        batch_size = batch["observations"].shape[0]
        z = jax.random.uniform(rng, shape=(batch_size, 1), minval=-49.3, maxval=0.0)
        
        # Get qz values from target qz_critic
        qzs = agent.target_qz_critic.apply_fn(
            {"params": agent.target_qz_critic.params},
            batch["observations"],
            z,
            batch["actions"],
        )
        qz = qzs.max(axis=0)
        
        # Get V(obs) from value network
        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        # Get Vc(obs) from safe_value network
        vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["observations"]
        )
        
        # Compute target: min(V - z, -Vc)
        z_squeezed = z.squeeze(-1)  # Shape (batch_size,)
        target_vz1 = jnp.minimum(v - z_squeezed, -vc)

        def vz_loss_fn(vz_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vz_val = agent.vz.apply_fn({"params": vz_params}, batch["observations"], z)
            vz_loss = expectile_loss(qz - vz_val, agent.cost_critic_hyperparam).mean() + 0.25 * (jnp.maximum(vz_val - target_vz1, 0*vz_val)).mean()

            return vz_loss, {"vz_loss": vz_loss, "vz": vz_val.mean(), "vz_max": vz_val.max(), "vz_min": vz_val.min(), "z_mean": z.mean()}

        grads, info = jax.grad(vz_loss_fn, has_aux=True)(agent.vz.params)
        vz = agent.vz.apply_gradients(grads=grads)

        agent = agent.replace(vz=vz)

        return agent, info
    
    def update_qz(agent, batch: DatasetDict, rng: jax.random.PRNGKey) -> Tuple[Agent, Dict[str, float]]:
        """Update qz_critic (Q-function with z). z follows dynamics: z_{t+1} = z_t - reward_t."""
        # Sample z_t from [-49.3, 0]
        batch_size = batch["observations"].shape[0]
        z = jax.random.uniform(rng, shape=(batch_size, 1), minval=-49.3, maxval=0.0)
        
        # Compute z_{t+1} = z_t - reward_t
        rewards = 0.1 * batch["rewards"].reshape(-1, 1) if batch["rewards"].ndim == 1 else 0.1 * batch["rewards"]
        next_z = (z - rewards)/0.99
        
        # Get h_t from batch (h_t = costs)
        h = batch["costs"].reshape(-1, 1) if batch["costs"].ndim == 1 else batch["costs"]
        
        # Get next vz for bootstrapping using next_z
        next_vz = agent.vz.apply_fn(
            {"params": agent.vz.params}, batch["next_observations"], next_z
        )
        
        # Compute target using h_t and min
        if agent.critic_type == "hj":
            # HJ-style update: Qz(s,z,a) = (1-γ)*(-h_t) + γ*min(-h_t, Vz(s',z'))
            neg_h = -h.squeeze(-1)
            neg_z = -z.squeeze(-1)
            qz_nonterminal = (1. - agent.discount) * neg_h + agent.discount * jnp.minimum(neg_h, next_vz)
            target_qz = qz_nonterminal * batch["masks"] + jnp.minimum(neg_h, neg_z) * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            # Standard Q-learning style: Qz(s,z,a) = -h + γ*Vz(s',z')
            neg_h = -h.squeeze(-1)
            target_qz = neg_h + agent.discount * batch["masks"] * next_vz
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

        def qz_critic_loss_fn(qz_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qzs = agent.qz_critic.apply_fn(
                {"params": qz_critic_params}, batch["observations"], z, batch["actions"]
            )
            qz_critic_loss = ((qzs - target_qz) ** 2).mean()

            return qz_critic_loss, {
                "qz_critic_loss": qz_critic_loss,
                "qz": qzs.mean(),
                "qz_max": qzs.max(),
                "qz_min": qzs.min(),
                "z_mean": z.mean(),
                "next_z_mean": next_z.mean(),
            }

        grads, info = jax.grad(qz_critic_loss_fn, has_aux=True)(agent.qz_critic.params)
        qz_critic = agent.qz_critic.apply_gradients(grads=grads)

        # Update target qz_critic
        target_qz_critic_params = optax.incremental_update(
            qz_critic.params, agent.target_qz_critic.params, agent.tau
        )
        target_qz_critic = agent.target_qz_critic.replace(params=target_qz_critic_params)

        agent = agent.replace(qz_critic=qz_critic, target_qz_critic=target_qz_critic)

        return agent, info
    
    def update_vz1(agent, batch: DatasetDict, rng: jax.random.PRNGKey) -> Tuple[Agent, Dict[str, float]]:
        """Update vz1 (value function with z) that learns min(V - z, -Vc) in supervised manner from V and Vc networks."""
        # Sample z_t from [-49.3, 0]
        batch_size = batch["observations"].shape[0]
        z = jax.random.uniform(rng, shape=(batch_size, 1), minval=-49.3, maxval=0.0)
        
        # Get V(obs) from value network
        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        # Get Vc(obs) from safe_value network
        vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["observations"]
        )
        
        # Compute target: min(V - z, -Vc)
        z_squeezed = z.squeeze(-1)  # Shape (batch_size,)
        target_vz1 = jnp.minimum(v - z_squeezed, -vc)

        def vz1_loss_fn(vz1_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vz1_val = agent.vz1.apply_fn({"params": vz1_params}, batch["observations"], z)
            # Supervised MSE loss
            vz1_loss = ((vz1_val - target_vz1) ** 2).mean()

            return vz1_loss, {"vz1_loss": vz1_loss, "vz1": vz1_val.mean(), "vz1_max": vz1_val.max(), "vz1_min": vz1_val.min(), "target_vz1_mean": target_vz1.mean()}

        grads, info = jax.grad(vz1_loss_fn, has_aux=True)(agent.vz1.params)
        vz1 = agent.vz1.apply_gradients(grads=grads)

        agent = agent.replace(vz1=vz1)

        return agent, info
    
    def update_qz1(agent, batch: DatasetDict, rng: jax.random.PRNGKey) -> Tuple[Agent, Dict[str, float]]:
        """Update qz1_critic (Q-function with z) that learns min(V - z, -Vc) in supervised manner from V and Vc networks."""
        # Sample z_t from [-49.3, 0]
        batch_size = batch["observations"].shape[0]
        z = jax.random.uniform(rng, shape=(batch_size, 1), minval=-49.3, maxval=0.0)
        
        # Get V(obs) from value network (use target for stability)
        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        # Get Vc(obs) from safe_value network
        vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["observations"]
        )
        
        # Compute target: min(V - z, -Vc)
        z_squeezed = z.squeeze(-1)  # Shape (batch_size,)
        target_qz1 = jnp.minimum(v - z_squeezed, -vc)

        def qz1_critic_loss_fn(qz1_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qz1s = agent.qz1_critic.apply_fn(
                {"params": qz1_critic_params}, batch["observations"], z, batch["actions"]
            )
            # Supervised MSE loss - all ensemble members learn the same target
            qz1_critic_loss = ((qz1s - target_qz1) ** 2).mean()

            return qz1_critic_loss, {
                "qz1_critic_loss": qz1_critic_loss,
                "qz1": qz1s.mean(),
                "qz1_max": qz1s.max(),
                "qz1_min": qz1s.min(),
                "target_qz1_mean": target_qz1.mean(),
                "v_mean": v.mean(),
                "neg_vc_mean": (-vc).mean(),
            }

        grads, info = jax.grad(qz1_critic_loss_fn, has_aux=True)(agent.qz1_critic.params)
        qz1_critic = agent.qz1_critic.apply_gradients(grads=grads)

        # Update target qz1_critic
        target_qz1_critic_params = optax.incremental_update(
            qz1_critic.params, agent.target_qz1_critic.params, agent.tau
        )
        target_qz1_critic = agent.target_qz1_critic.replace(params=target_qz1_critic_params)

        agent = agent.replace(qz1_critic=qz1_critic, target_qz1_critic=target_qz1_critic)

        return agent, info
    
    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

            
        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    def opt_value_func_calc_jax_batch(agent, Vr, Vh, z_range, num_z):
        Vr_arr = jnp.asarray(Vr)
        Vh_arr = jnp.asarray(Vh)

        # remember if both were scalars so we can return scalar float
        scalar_input = (Vr_arr.ndim == 0) and (Vh_arr.ndim == 0)

        # broadcast to a common batch shape and flatten to (batch,)
        Vr_b, Vh_b = jnp.broadcast_arrays(Vr_arr, Vh_arr)
        Vr_flat = Vr_b.ravel()
        Vh_flat = Vh_b.ravel()

        # reversed z list (high to low)
        z_list = jnp.linspace(z_range[1], z_range[0], num_z)

        # compute V for every (batch, z) pair: shape (batch, num_z)
        V = jnp.minimum(Vh_flat[:, None], Vr_flat[:, None] - z_list[None, :])

        # find first z index (smallest index because z_list is reversed) where V >= 0
        cond = V >= 0.0               # boolean matrix
        found_any = jnp.any(cond, axis=1)
        first_idx = jnp.argmax(cond, axis=1)   # returns index of first True (or 0 if none)

        # select z or sentinel
        z_opt_flat = jnp.where(found_any, z_list[first_idx], -3000.0)

        if scalar_input:
            return float(z_opt_flat[0])
        return z_opt_flat

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        # Flow matching training: sample time t ~ U(0, 1) and interpolate
        batch_size = batch['actions'].shape[0]
        
        # Sample random noise x_0 ~ N(0, I)
        key, rng = jax.random.split(rng, 2)
        x_0 = jax.random.normal(key, (batch_size, agent.act_dim))
        
        # Target actions x_1
        x_1 = batch['actions']
        
        # Sample time t ~ U(0, 1)
        key, rng = jax.random.split(rng, 2)
        time = jax.random.uniform(key, (batch_size, 1))
        
        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - time) * x_0 + time * x_1
        
        # Target velocity: v = x_1 - x_0
        target_velocity = x_1 - x_0

        key, rng = jax.random.split(rng, 2)
        
        '''
        reward adv
        '''
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        '''
        cost reward
        '''
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        qc = qcs.max(axis=0)

        vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"]
            )

        scale = 0.05

        if agent.critic_type == "qc":
            qc = qc - agent.qc_thres
            vc = vc - agent.qc_thres

        if agent.actor_objective == "feasibility":
            eps = 0. if agent.critic_type != 'qc' else 0.
            
            # unsafe_condition = jnp.where( vc >  0. - eps, 1, 0)
            # safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc<=0. - eps, 1, 0)

            z_opt = agent.opt_value_func_calc_jax_batch(v, -vc, (-1100.0, 3100.0), 1000)   #SafeVelocity
            # z_opt = agent.opt_value_func_calc_jax_batch(v, -vc, (-100.0, 100.0), 200)  #CarNavigation
            # z_opt = agent.opt_value_func_calc_jax_batch(v, -vc, (-10 * 4.93, 0.0), 400)    #boat

            unsafe_condition = jnp.where( -vc < (v - z_opt)/scale, 1, 0)
            safe_condition = jnp.where(-vc >= (v - z_opt)/scale, 1, 0) #* jnp.where(qc<=0. - eps, 1, 0)
            
            cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
            reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
            
            unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub) ## ignore vc >0, qc>vc
            safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
            
            weights = unsafe_weights + safe_weights #jnp.clip(cost_exp_adv, 0, agent.cost_ub)
        elif agent.actor_objective == "bc":
            weights = jnp.ones(qc.shape)
        else:
            raise ValueError(f'Invalid actor objective: {agent.actor_objective}')
        
        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # Predict velocity field
            velocity_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       x_t,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            # Flow matching loss: MSE between predicted and target velocity
            actor_loss = (((velocity_pred - target_velocity) ** 2).sum(axis = -1) * weights).mean()
            
            return actor_loss, {'actor_loss': actor_loss, 'weights' : weights.mean()}
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)

        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        N = 8
        # print(">>>>>>>>>>>>>>>>>N:", N)
        observations = jnp.expand_dims(observations, axis = 0).repeat(N, axis = 0)
        # print("observations shape:", observations.shape)
        score_params = self.target_score_model.params
        
        # Use flow matching sampler (Euler integration)
        actions, rng = flow_matching_sampler(
            self.score_model.apply_fn,
            score_params,
            self.T,  # num_steps for integration
            rng,
            self.act_dim,
            observations,
            clip_sampler=self.clip_sampler,
            training=False
        )
        
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions)

        if self.critic_type == "qc":
            qcs = qcs - self.qc_thres
        # print("qs:", qs)
        z_opt = self.opt_value_func_calc_jax_batch(qs, -qcs, (-1100.0, 3100.0), 1000)   #SafeVelocity
        # z_opt = self.opt_value_func_calc_jax_batch(qs, -qcs, (-100.0, 100.0), 200)  #CarNavigation
        # z_opt = self.opt_value_func_calc_jax_batch(qs, -qcs, (-400 * 4.93, 0.0), 400)    #boat
        scale = 0.05

        if self.extract_method == 'safemaxq':
            # safe_indices = jnp.where(qcs < -0.03)[0]
            # if len(safe_indices) == 0:
            #     idx = jnp.argmin(qcs)
            # else:
            #     safe_qs = qs[safe_indices]
            #     max_safe_idx = jnp.argmax(safe_qs)
            #     idx = safe_indices[max_safe_idx]
            Vhat = jnp.minimum( -qcs , (qs - z_opt)/scale )
            idx = jnp.argmax(Vhat)
        elif self.extract_method == 'maxq':
            idx = jnp.argmax(qs)
        elif self.extract_method == 'minqc':
            idx = jnp.argmin(qcs)
        else:
            raise ValueError(f'Invalid extract_method: {self.extract_method}')
        action = actions[idx]
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    
    def actor_loss_no_grad(agent, batch: DatasetDict):
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        
        batch_size = batch['actions'].shape[0]
        
        # Sample random noise x_0 ~ N(0, I)
        x_0 = jax.random.normal(key, (batch_size, agent.act_dim))
        
        # Target actions x_1
        x_1 = batch['actions']
        
        # Sample time t ~ U(0, 1)
        key, rng = jax.random.split(rng, 2)
        time = jax.random.uniform(key, (batch_size, 1))
        
        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - time) * x_0 + time * x_1
        
        # Target velocity: v = x_1 - x_0
        target_velocity = x_1 - x_0

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            velocity_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       x_t,
                                       time,
                                       training=False)
            
            actor_loss = (((velocity_pred - target_velocity) ** 2).sum(axis = -1)).mean()

            return actor_loss, {'actor_loss': actor_loss}

        _, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        new_agent = agent.replace(rng=rng)

        return new_agent, info
    

    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def eval_loss(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.actor_loss_no_grad(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        def slice(x):
            return x[:256]

        new_agent = self
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)

        return new_agent, {**critic_info, **value_info}

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch['observations'].shape[0]/2)

        def first_half(x):
            return x[:batch_size]
        
        def second_half(x):
            return x[batch_size:]
        
        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)
        new_agent, safe_critic_info = new_agent.update_vc(mini_batch)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch)
        
        # Update Qz and Vz (z-augmented value functions)
        rng, qz_key, vz_key, qz1_key, vz1_key = jax.random.split(new_agent.rng, 5)
        new_agent = new_agent.replace(rng=rng)
        new_agent, qz_info = new_agent.update_qz(mini_batch, qz_key)
        new_agent, vz_info = new_agent.update_vz(mini_batch, vz_key)
        
        # Update Qz1 and Vz1 (learns min(V - z, -Vc))
        new_agent, qz1_info = new_agent.update_qz1(mini_batch, qz1_key)
        new_agent, vz1_info = new_agent.update_vz1(mini_batch, vz1_key)
        
        # Print losses for monitoring
        # print(f"Losses - vz: {vz_info['vz_loss']:.4f}, qz: {qz_info['qz_critic_loss']:.4f}, vz1: {vz1_info['vz1_loss']:.4f}, qz1: {qz1_info['qz1_critic_loss']:.4f}")

        return new_agent, {**actor_info, **critic_info, **value_info, **safe_critic_info, **safe_value_info, **qz_info, **vz_info, **qz1_info, **vz1_info}
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent