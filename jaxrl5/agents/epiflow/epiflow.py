"""EpiFlow: Epigraph-Guided Flow Matching for safe offline RL.

    M. Tayal, M. Tayal, R. Prakash. "EpiFlow: Bridging Safety and Performance in
    Offline Reinforcement Learning via Epigraph-Guided Flow Matching." NeurIPS 2026.

Built on the SafeFQL codebase (https://github.com/tayalmanan28/safe-fql): the
reward / cost critics, the flow actors and the training loop follow SafeFQL; the
epigraph feasibility value function and the z*-conditioned actor objective are
EpiFlow's additions.

Conventions (as in SafeFQL): ``batch["costs"]`` is the negated safety function,
c(x) = -l(x), so c > 0 inside the failure set. V_c(x) is the HJ cost value and the
safety value of the paper is V_s(x) = -V_c(x). The epigraph value functions are
expressed in the paper's convention (V_hat >= 0 <=> feasible).

  Reward values      Q_r(x,a), V_r(x)          IQL / expectile        (Eq. 30-31)
  Safety values      Q_c(x,a), V_c(x)          HJ backup + expectile  (Eq. 32-33)
  Epigraph values    Q_hat(x,z,a), V_hat(x,z)  with z' = (z - r)/gamma (Eq. 4)
      Q_hat target   (1-gamma) l + gamma min{l, V_hat(x',z')},  min{l, r - z} at terminals
      V_hat loss     L_tau(Q_hat - V_hat) + lambda max(0, V_hat - min(V_r - z, V_s))
                                                                      (Eq. 10-12)
  Budget             z*(x) = sup{z : V_hat(x,z) >= 0}, by bisection   (Eq. 13)
  Flow teacher       conditional flow matching on dataset actions     (Eq. 15)
  One-step actor     -Q_hat(x, z*(x), mu(x,eta)) + alpha * distillation  (Eq. 16-17)
"""
import os
from functools import partial
from typing import Dict, Sequence, Tuple

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from flax.training.train_state import TrainState
from flax import struct

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    StateActionValueWithZ,
    StateValue,
    StateValueWithZ,
)


# =====================================================================
# Helper losses (from SafeFQL)
# =====================================================================

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


# =====================================================================
# Flow Vector Field network (from SafeFQL)
# =====================================================================

class FlowVectorField(nn.Module):
    """Vector field network for flow matching.

    Used for both the BC flow teacher (with times) and the one-step actor (without times).
    """
    hidden_dims: Sequence[int]
    action_dim: int
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, actions, times=None, training=False):
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        x = MLP(
            hidden_dims=tuple(list(self.hidden_dims) + [self.action_dim]),
            activations=nn.gelu,
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
        )(inputs, training=training)
        return x


# =====================================================================
# EpiFlow Agent
# =====================================================================

class EpiFlow(Agent):
    # ---- Flow-based actor (SafeFQL) ----
    actor_bc_flow: TrainState         # flow behaviour teacher mu_zeta
    actor_onestep_flow: TrainState    # one-step actor mu_omega

    # ---- Reward critics (SafeFQL) ----
    critic: TrainState                # Q_r(x,a)
    target_critic: TrainState
    value: TrainState                 # V_r(x)

    # ---- Cost / safety critics (SafeFQL) ----
    safe_critic: TrainState           # Q_c(x,a)
    safe_target_critic: TrainState
    safe_value: TrainState            # V_c(x) = -V_s(x)

    # ---- Epigraph feasibility value function (EpiFlow) ----
    epi_critic: TrainState            # Q_hat(x,z,a)
    target_epi_critic: TrainState
    epi_value: TrainState             # V_hat(x,z)

    # ---- Continuous hyper-parameters ----
    discount: float
    tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    qc_thres: float
    alpha: float                      # distillation coefficient
    reg_weight: float                 # lambda, decomposition regulariser
    z_min: float
    z_max: float

    # ---- Static hyper-parameters ----
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    flow_steps: int = struct.field(pytree_node=False)
    normalize_q_loss: bool = struct.field(pytree_node=False)
    bisection_steps: int = struct.field(pytree_node=False)
    normalize_z: bool = struct.field(pytree_node=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        z_min: float,
        z_max: float,
        # --- actor (FQL) ---
        actor_lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
        actor_layer_norm: bool = False,
        alpha: float = 0.1,
        flow_steps: int = 10,
        normalize_q_loss: bool = True,
        # --- critics (SafeFQL) ---
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        value_layer_norm: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.9,
        cost_critic_hyperparam: float = 0.9,
        critic_objective: str = "expectile",
        critic_type: str = "hj",
        cost_limit: float = 10.0,
        env_max_steps: int = 1000,
        # --- epigraph (EpiFlow) ---
        reg_weight: float = 0.25,
        bisection_steps: int = 20,
        normalize_z: bool = True,
        # --- evaluation ---
        N: int = 1,
    ):
        if critic_type != "hj":
            raise ValueError("EpiFlow requires critic_type='hj' (state-wise safety constraint).")

        rng = jax.random.PRNGKey(seed)
        rng, bc_key, os_key, c_key, v_key, sc_key, sv_key, ec_key, ev_key = jax.random.split(rng, 9)

        observations = observation_space.sample()
        actions = action_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount ** env_max_steps) / (
            1 - discount
        ) / env_max_steps

        obs_b = jnp.expand_dims(jnp.array(observations), axis=0)
        act_b = jnp.expand_dims(jnp.array(actions), axis=0)
        time_b = jnp.zeros((1, 1))
        z_b = jnp.zeros((1, 1))

        frozen = optax.GradientTransformation(lambda _: None, lambda _: None)

        # ---- Flow actors (FQL-style) ----
        flow_def = FlowVectorField(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            use_layer_norm=actor_layer_norm,
        )
        actor_bc_flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_def.init(bc_key, obs_b, act_b, time_b)["params"],
            tx=optax.adam(learning_rate=actor_lr),
        )
        actor_onestep_flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_def.init(os_key, obs_b, act_b)["params"],
            tx=optax.adam(learning_rate=actor_lr),
        )

        # ---- Reward critics ----
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_def = Ensemble(partial(StateActionValue, base_cls=critic_base_cls), num=num_qs)
        critic_opt = optax.adam(learning_rate=critic_lr)
        critic_params = critic_def.init(c_key, obs_b, act_b)["params"]
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=critic_opt)
        target_critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=frozen)

        # ---- Cost critics ----
        safe_critic_params = critic_def.init(sc_key, obs_b, act_b)["params"]
        safe_critic = TrainState.create(apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_opt)
        safe_target_critic = TrainState.create(apply_fn=critic_def.apply, params=safe_critic_params, tx=frozen)

        # ---- Value networks ----
        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True,
                                 use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_opt = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_def.init(v_key, obs_b)["params"], tx=value_opt)
        safe_value = TrainState.create(apply_fn=value_def.apply,
                                       params=value_def.init(sv_key, obs_b)["params"], tx=value_opt)

        # ---- Epigraph critics ----
        epi_critic_def = Ensemble(partial(StateActionValueWithZ, base_cls=critic_base_cls), num=num_qs)
        epi_critic_params = epi_critic_def.init(ec_key, obs_b, z_b, act_b)["params"]
        epi_critic = TrainState.create(apply_fn=epi_critic_def.apply, params=epi_critic_params, tx=critic_opt)
        target_epi_critic = TrainState.create(apply_fn=epi_critic_def.apply, params=epi_critic_params, tx=frozen)

        epi_value_def = StateValueWithZ(base_cls=value_base_cls)
        epi_value = TrainState.create(apply_fn=epi_value_def.apply,
                                      params=epi_value_def.init(ev_key, obs_b, z_b)["params"],
                                      tx=value_opt)

        return cls(
            actor=None,  # base-class attribute (unused)
            rng=rng,
            actor_bc_flow=actor_bc_flow,
            actor_onestep_flow=actor_onestep_flow,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            epi_critic=epi_critic,
            target_epi_critic=target_epi_critic,
            epi_value=epi_value,
            discount=discount,
            tau=tau,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            qc_thres=qc_thres,
            alpha=alpha,
            reg_weight=reg_weight,
            z_min=float(z_min),
            z_max=float(z_max),
            critic_objective=critic_objective,
            critic_type=critic_type,
            act_dim=action_dim,
            N=N,
            flow_steps=flow_steps,
            normalize_q_loss=normalize_q_loss,
            bisection_steps=bisection_steps,
            normalize_z=normalize_z,
        )

    # ==================================================================
    # Reward / cost critic updates (SafeFQL)
    # ==================================================================

    def update_v(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update reward value V_r(x) via expectile regression against Q_r(x,a)."""
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params):
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            if agent.critic_objective == "expectile":
                loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f"Invalid critic objective: {agent.critic_objective}")
            return loss, {"value_loss": loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        return agent.replace(value=value), info

    def update_q(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update reward Q_r(x,a) via TD learning with V_r(x')."""
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params):
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        return agent.replace(critic=critic, target_critic=target_critic), info

    def update_vc(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update cost value V_c(x) via safe expectile regression against Q_c(x,a)."""
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params):
            vc = agent.safe_value.apply_fn(
                {"params": safe_value_params}, batch["observations"]
            )
            loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()
            return loss, {
                "safe_value_loss": loss,
                "vc": vc.mean(),
                "vc_max": vc.max(),
                "vc_min": vc.min(),
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update cost Q_c(x,a) via the discounted HJ backup with V_c(x')."""
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        qc_nonterminal = (1.0 - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_vc)
        target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])

        def safe_critic_loss_fn(safe_critic_params):
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"],
                batch["actions"],
            )
            loss = ((qcs - target_qc) ** 2).mean()
            return loss, {
                "safe_critic_loss": loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean(),
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        return agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic), info

    # ==================================================================
    # Epigraph feasibility value function (EpiFlow)
    # ==================================================================

    def _z_input(agent, z):
        """Rescale z from [z_min, z_max] to [-1, 1] before it enters a network."""
        if not agent.normalize_z:
            return z
        mid = 0.5 * (agent.z_max + agent.z_min)
        half = 0.5 * (agent.z_max - agent.z_min)
        return (z - mid) / half

    def epi_v(agent, observations, z, params=None):
        """V_hat(x, z), z of shape (batch, 1)."""
        params = agent.epi_value.params if params is None else params
        return agent.epi_value.apply_fn({"params": params}, observations, agent._z_input(z))

    def epi_q(agent, observations, z, actions, params=None):
        """min over the ensemble of Q_hat(x, z, a)."""
        params = agent.epi_critic.params if params is None else params
        qs = agent.epi_critic.apply_fn({"params": params}, observations, agent._z_input(z), actions)
        return qs.min(axis=0)

    def optimal_z(agent, observations):
        """z*(x) = sup{z in [z_min, z_max] : V_hat(x, z) >= 0} by bisection (Eq. 13).

        If no threshold is feasible, z* clamps to z_min. Returns shape (batch, 1).
        """
        batch = observations.shape[0]
        lo = jnp.full((batch, 1), agent.z_min)
        hi = jnp.full((batch, 1), agent.z_max)
        top_feasible = (agent.epi_v(observations, hi) >= 0.0)[:, None]

        def body(_, carry):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            feasible = (agent.epi_v(observations, mid) >= 0.0)[:, None]
            return jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)

        lo, _ = jax.lax.fori_loop(0, agent.bisection_steps, body, (lo, hi))
        return jnp.where(top_feasible, hi, lo)

    def update_epigraph(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update Q_hat(x,z,a) and V_hat(x,z) (Algorithm 1)."""
        rng, z_key = jax.random.split(agent.rng)
        obs = batch["observations"]
        batch_size = obs.shape[0]

        z = jax.random.uniform(z_key, (batch_size, 1), minval=agent.z_min, maxval=agent.z_max)
        next_z = (z - batch["rewards"][:, None]) / agent.discount
        l = -batch["costs"]
        zs = z[:, 0]

        # ---- Q_hat: discounted HJ form of y_Q = min{l(x), V_hat(x', z')}.
        #      At a terminal transition the return from x is the last reward r, so the
        #      target is min{l(x), r - z} (Eq. 3 with no future after x').
        next_vz = agent.epi_v(batch["next_observations"], next_z)
        qz_nonterminal = (1.0 - agent.discount) * l + agent.discount * jnp.minimum(l, next_vz)
        qz_terminal = jnp.minimum(l, batch["rewards"] - zs)
        target_qz = qz_nonterminal * batch["masks"] + qz_terminal * (1 - batch["masks"])

        def epi_critic_loss_fn(params):
            qzs = agent.epi_critic.apply_fn({"params": params}, obs, agent._z_input(z), batch["actions"])
            loss = ((qzs - target_qz) ** 2).mean()
            return loss, {"epi_critic_loss": loss, "qz": qzs.mean(), "z_mean": z.mean()}

        grads, qz_info = jax.grad(epi_critic_loss_fn, has_aux=True)(agent.epi_critic.params)
        epi_critic = agent.epi_critic.apply_gradients(grads=grads)
        target_epi_critic = agent.target_epi_critic.replace(params=optax.incremental_update(
            epi_critic.params, agent.target_epi_critic.params, agent.tau))

        # ---- V_hat: expectile distillation + decomposition regulariser
        qz = agent.epi_q(obs, z, batch["actions"], params=agent.target_epi_critic.params)
        v = agent.value.apply_fn({"params": agent.value.params}, obs)
        vc = agent.safe_value.apply_fn({"params": agent.safe_value.params}, obs)
        upper = jnp.minimum(v - zs, -vc)                # min(V_r - z, V_s), Eq. (6)

        def epi_value_loss_fn(params):
            vz = agent.epi_v(obs, z, params=params)
            distill = expectile_loss(qz - vz, agent.cost_critic_hyperparam).mean()
            reg = jnp.maximum(vz - upper, 0.0).mean()
            loss = distill + agent.reg_weight * reg
            return loss, {
                "epi_value_loss": loss,
                "epi_value_distill": distill,
                "epi_value_reg": reg,
                "vz": vz.mean(),
                "frac_feasible": (vz >= 0).mean(),
            }

        grads, vz_info = jax.grad(epi_value_loss_fn, has_aux=True)(agent.epi_value.params)
        epi_value = agent.epi_value.apply_gradients(grads=grads)

        return agent.replace(epi_critic=epi_critic, target_epi_critic=target_epi_critic,
                             epi_value=epi_value, rng=rng), {**qz_info, **vz_info}

    # ==================================================================
    # FQL-style flow actor updates
    # ==================================================================

    def _compute_flow_actions(agent, observations, noises):
        """Run the BC flow model for `flow_steps` Euler steps (no gradient)."""
        actions = noises
        for i in range(agent.flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / agent.flow_steps)
            vels = agent.actor_bc_flow.apply_fn(
                {"params": agent.actor_bc_flow.params},
                observations,
                actions,
                t,
            )
            actions = actions + vels / agent.flow_steps
        return jnp.clip(actions, -1, 1)

    def update_bc_flow(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update the flow behaviour teacher via flow matching (Eq. 15)."""
        rng = agent.rng
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        batch_size = batch["actions"].shape[0]
        x_0 = jax.random.normal(x_rng, (batch_size, agent.act_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        def bc_flow_loss_fn(bc_flow_params):
            pred = agent.actor_bc_flow.apply_fn(
                {"params": bc_flow_params},
                batch["observations"],
                x_t,
                t,
            )
            loss = ((pred - vel) ** 2).sum(axis=-1).mean()
            return loss, {"bc_flow_loss": loss}

        grads, info = jax.grad(bc_flow_loss_fn, has_aux=True)(agent.actor_bc_flow.params)
        actor_bc_flow = agent.actor_bc_flow.apply_gradients(grads=grads)
        return agent.replace(actor_bc_flow=actor_bc_flow, rng=rng), info

    def update_onestep_flow(agent, batch: DatasetDict) -> Tuple["EpiFlow", Dict[str, float]]:
        """Update the one-step actor: -Q_hat(x, z*(x), a) + alpha * distillation (Eq. 17)."""
        rng = agent.rng
        rng, noise_rng = jax.random.split(rng)

        obs = batch["observations"]
        noises = jax.random.normal(noise_rng, (obs.shape[0], agent.act_dim))
        target_flow_actions = agent._compute_flow_actions(obs, noises)
        z_star = jax.lax.stop_gradient(agent.optimal_z(obs))

        def onestep_loss_fn(onestep_params):
            actor_actions = agent.actor_onestep_flow.apply_fn(
                {"params": onestep_params}, obs, noises,
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            q = agent.epi_q(obs, z_star, jnp.clip(actor_actions, -1, 1))
            q_loss = -q.mean()
            if agent.normalize_q_loss:
                lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
                q_loss = lam * q_loss

            total = agent.alpha * distill_loss + q_loss
            return total, {
                "onestep_total_loss": total,
                "distill_loss": distill_loss,
                "q_loss": q_loss,
                "actor_qz": q.mean(),
                "z_star": z_star.mean(),
            }

        grads, info = jax.grad(onestep_loss_fn, has_aux=True)(agent.actor_onestep_flow.params)
        actor_onestep_flow = agent.actor_onestep_flow.apply_gradients(grads=grads)
        return agent.replace(actor_onestep_flow=actor_onestep_flow, rng=rng), info

    # ==================================================================
    # Combined update (SafeFQL training loop)
    # ==================================================================

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch["observations"].shape[0] / 2)

        def first_half(x):
            return x[:batch_size]

        def second_half(x):
            return x[batch_size:]

        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        # --- Flow actor updates (two gradient steps each) ---
        new_agent, _ = new_agent.update_bc_flow(first_batch)
        new_agent, bc_flow_info = new_agent.update_bc_flow(second_batch)
        new_agent, _ = new_agent.update_onestep_flow(first_batch)
        new_agent, onestep_info = new_agent.update_onestep_flow(second_batch)

        # --- Critic updates (on mini-batch) ---
        def slice_mini(x):
            return x[:256]

        mini_batch = jax.tree_util.tree_map(slice_mini, batch)
        new_agent, v_info = new_agent.update_v(mini_batch)
        new_agent, q_info = new_agent.update_q(mini_batch)
        new_agent, vc_info = new_agent.update_vc(mini_batch)
        new_agent, qc_info = new_agent.update_qc(mini_batch)
        new_agent, epi_info = new_agent.update_epigraph(mini_batch)

        return new_agent, {
            **bc_flow_info,
            **onestep_info,
            **v_info,
            **q_info,
            **vc_info,
            **qc_info,
            **epi_info,
        }

    # ==================================================================
    # Evaluation
    # ==================================================================

    def eval_actions(self, observations: jnp.ndarray):
        """Single forward pass of the one-step actor."""
        rng = self.rng
        assert len(observations.shape) == 1

        observations = jax.device_put(observations)
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, (self.act_dim,))

        action = self.actor_onestep_flow.apply_fn(
            {"params": self.actor_onestep_flow.params},
            observations,
            noise,
        )
        action = jnp.clip(action, -1, 1)
        return np.array(action), self.replace(rng=rng)

    # ==================================================================
    # Serialization
    # ==================================================================

    def save(self, modeldir, save_time):
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        file_name = f"model{save_time}.pickle"
        state_dict = flax.serialization.to_state_dict(self)
        with open(os.path.join(modeldir, file_name), "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, model_location):
        with open(model_location, "rb") as f:
            pkl_file = pickle.load(f)
        return flax.serialization.from_state_dict(target=self, state=pkl_file)
