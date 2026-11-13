from typing import Dict

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_pr(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens = [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}


def evaluate_br(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens = [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}


# =====================================================================
# Evaluation protocol of the paper (Section 4), used by
# launcher/examples/evaluate_policy.py:
#   * a fixed set of initial states shared across methods (episode i of a DSRL
#     task is reset with seed=i; Boat Navigation uses 500 fixed safe states),
#   * DSRL normalisation R_norm = (R - r_min) / (r_max - r_min) and
#     C_norm = C / kappa with cost limit kappa = 10,
#   * optional Gaussian action perturbations (Appendix C.2).
# =====================================================================

COST_LIMIT = 10.0
BOAT_REWARD_RANGE = (-1963.26, -167.80)   # min / max episode return of the Boat dataset


def _summarise(rets, costs, lens, r_range):
    rets, costs = np.asarray(rets), np.asarray(costs)
    info = {
        "return": float(rets.mean()),
        "return_std": float(rets.std()),
        "cost": float(costs.mean()),
        "cost_std": float(costs.std()),
        "safety_rate": float((costs == 0).mean()),
        "episode_len": float(np.mean(lens)),
        "num_episodes": int(len(rets)),
    }
    if r_range is not None:
        r_min, r_max = r_range
        info["normalized_return"] = float((rets.mean() - r_min) / (r_max - r_min))
        info["normalized_cost"] = float(costs.mean() / COST_LIMIT)
    return info


def evaluate_fixed(agent, env: gym.Env, num_episodes: int, action_noise: float = 0.0,
                   seed: int = 0) -> Dict[str, float]:
    """DSRL tasks: episode i is reset with seed=i."""
    noise_rng = np.random.default_rng(seed)
    rets, costs, lens = [], [], []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        obs, _ = env.reset(seed=ep)
        ret, cost, length = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            if action_noise > 0:
                action = np.clip(action + noise_rng.normal(0.0, action_noise, action.shape), -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            cost += info["cost"]
            length += 1
            if terminated or truncated:
                break
        rets.append(ret)
        costs.append(cost)
        lens.append(length)
    return _summarise(rets, costs, lens, (env.min_episode_reward, env.max_episode_reward))


def boat_initial_states(env, num_states: int = 500, seed: int = 0) -> np.ndarray:
    """Fixed set of safe initial states for Boat Navigation."""
    rng = np.random.default_rng(seed)
    low, high = env.observation_space.low, env.observation_space.high
    states = []
    while len(states) < num_states:
        s = rng.uniform(low, high)
        if not env.check_violation(s).item():
            states.append(s)
    return np.asarray(states, dtype=np.float32)


def evaluate_boat_fixed(agent, env, num_episodes: int, action_noise: float = 0.0,
                        seed: int = 0) -> Dict[str, float]:
    """Boat Navigation from fixed safe initial states."""
    noise_rng = np.random.default_rng(seed)
    init_states = boat_initial_states(env, num_episodes, seed)
    rets, costs, lens = [], [], []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset(state=np.array(init_states[ep]))
        ret, cost, length = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            if action_noise > 0:
                action = np.clip(action + noise_rng.normal(0.0, action_noise, action.shape), -1, 1)
            obs, reward, done, info = env.step(action)
            ret += reward
            cost += info["cost"]
            length += 1
            if done or length == env._max_episode_steps:
                break
        rets.append(ret)
        costs.append(cost)
        lens.append(length)
    return _summarise(rets, costs, lens, BOAT_REWARD_RANGE)
