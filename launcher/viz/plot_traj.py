#!/usr/bin/env python3
"""
plot_traj.py

Usage examples:
  python plot_traj.py --results_dir ./results/OfflineAnt... --checkpoint latest --episodes 8 --out all_traj.png
  python plot_traj.py --episodes 10 --out all_traj.png         # uses random policy
  python plot_traj.py --episodes 10 --init_states_csv init.csv
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import random
import math
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path if you run from results folder
import sys
sys.path.append('.')

import gymnasium as gym
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot

# --- helper utilities -------------------------------------------------

def find_model_ckpt(results_dir: Path, checkpoint: str = 'latest') -> Path:
    """Prefer model*.pickle files. Return Path or None."""
    if not results_dir.exists():
        return None
    if checkpoint == 'latest':
        candidates = sorted(results_dir.glob('model*.pickle'))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        # fallback: any file
        allc = list(results_dir.glob('*'))
        return max(allc, key=lambda p: p.stat().st_mtime) if allc else None
    else:
        p = results_dir / checkpoint
        if p.exists():
            return p
        # try model{checkpoint}.pickle
        p2 = results_dir / f'model{checkpoint}.pickle'
        if p2.exists():
            return p2
        p3 = results_dir / f'model_{checkpoint}.pickle'
        if p3.exists():
            return p3
        # try glob fallback
        candidates = sorted(results_dir.glob('model*.pickle'))
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


# simple safe-loading for Flax-serialized pickle saved by DiffCBF.save
# returns a deserialized dict or raw object
def inspect_pickle_file(p: Path):
    try:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Failed to load pickle file {p}: {e}")
        return None


# A small wrapper around the agent API used in training: .eval_actions(obs) -> (action, agent) or action
# We will attempt to call eval_actions, falling back to sample_actions or to random action
def get_action_from_agent(agent, obs: np.ndarray):
    # obs must be 1D array (PointRobot _get_obs returns length 11)
    try:
        out = agent.eval_actions(obs)
    except Exception as e:
        try:
            out = agent.sample_actions(obs)
        except Exception as e2:
            # random fallback
            return None, agent
    # out may be (action, agent) or just action
    if isinstance(out, tuple) and len(out) >= 1:
        action = out[0]
        # second element might be new agent; if so, update
        if len(out) >= 2 and hasattr(out[1], '__class__'):
            return np.asarray(action).squeeze(), out[1]
        else:
            return np.asarray(action).squeeze(), agent
    else:
        return np.asarray(out).squeeze(), agent


# Plot helpers
def draw_env_map(ax, env: PointRobot):
    # hazards
    for hazard_pos in env.hazard_position_list:
        circle = plt.Circle((hazard_pos[0], hazard_pos[1]), hazard_pos[2], color='k', fill=False, linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
    # goal
    ax.add_patch(plt.Circle((env.goal_position[0], env.goal_position[1]), env.goal_size, color=(0.35,0.66,0.35), alpha=0.6))
    ax.set_xlim([-3,2])
    ax.set_ylim([-2,2])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Agent trajectories')


# Helpers for initial-state handling and cost extraction
import pandas as pd
def load_init_states_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    return np.array(df.values, dtype=float)[:, :2]


# --- main rollout + plotting -------------------------------------------------

def run_and_plot(results_dir: Path = None, checkpoint: str = 'latest', episodes: int = 8, seed: int = None, out: str = 'all_trajectories.png', per_episode_dir: str = None, render: bool = False, init_states_csv: str = None):
    # seed
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    random.seed(seed)
    np.random.seed(seed)

    # build env
    env = BoatRobot(seed=seed)

    # load agent if requested
    agent = None
    ckpt_path = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        cfg_path = results_dir / 'config.json'
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = None
        ckpt_path = find_model_ckpt(results_dir, checkpoint)
        if ckpt_path is not None:
            print(f"Found checkpoint: {ckpt_path}")
            data = inspect_pickle_file(ckpt_path)
            if cfg is not None and 'agent_kwargs' in cfg and 'model_cls' in cfg['agent_kwargs']:
                # attempt to recreate the agent using the same procedure as training
                try:
                    # local import delayed to avoid heavy deps if not needed
                    import jaxrl5.agents as agents_module
                    agent_kwargs = dict(cfg.get('agent_kwargs', {}))
                    model_cls = agent_kwargs.pop('model_cls')
                    # find class
                    if hasattr(agents_module, model_cls):
                        agent_cls = getattr(agents_module, model_cls)
                    else:
                        # try globals
                        agent_cls = globals().get(model_cls)
                    # pass env action/obs spaces and env_max_steps if required
                    env_max_steps = getattr(env, '_max_episode_steps', 300)
                    agent_kwargs['env_max_steps'] = env_max_steps
                    if hasattr(agent_cls, 'create'):
                        agent = agent_cls.create(seed, env.observation_space, env.action_space, **agent_kwargs)
                    else:
                        agent = agent_cls(seed, env.observation_space, env.action_space, **agent_kwargs)
                    # now call agent.load if available and assign result
                    if hasattr(agent, 'load'):
                        loaded = agent.load(str(ckpt_path))
                        if loaded is None:
                            print('Agent.load returned None (assumed in-place load)')
                        else:
                            agent = loaded
                    else:
                        print('Agent has no load method; using random policy')
                except Exception as e:
                    print('Failed to recreate agent from config:', e)
                    agent = None
            else:
                print('No config.json or model_cls; will not try to re-create agent automatically')
        else:
            print('No checkpoint found; running random policy')

    # fallback: if no agent, use random policy
    use_random = agent is None
    if use_random:
        print('Using random policy for rollouts')

    # load initial states CSV if provided
    init_states = None
    if init_states_csv is not None:
        # init_states = load_init_states_from_csv(init_states_csv)
        # init_states = init_states[:100]
        init_states = np.array([
            [-2.58, 0.67],
            [-2.44, -1.16],
        ])
        if init_states is None:
            print("Warning: failed to load init states CSV; proceeding with normal resets.")
        else:
            print(f"Loaded {len(init_states)} initial states from CSV.")

    # prepare plotting & metrics
    all_trajectories = []
    per_episode_metadata = []

    returns = []
    costs = []
    comp_time = []
    episodes_with_violation = 0

    # episodes = len(init_states)
    # episodes = args.episodes
    # print("Episodes to run:", episodes)
    for ep in range(episodes):
    # for ep in [28, 69]:
        # choose initial state if CSV provided (wrap-around)
        chosen_xy = None
        if init_states is not None:
            chosen_xy = init_states[ep]

        # reset env and attempt to set initial state if provided
        obs = env.reset(state=chosen_xy)
        print(f"Episode {ep+1}/{episodes} starting from state: {obs}")

        # begin episode
        traj = []
        # assume observation contains x,y at indices 0 and 1
        traj.append((float(obs[0]), float(obs[1])))
        
        done = False
        steps = 0
        current_agent = agent

        ep_return = 0.0
        ep_cost = 0.0
        any_violation = False
        start_time = time.time()
        while not done and steps < getattr(env, '_max_episode_steps', 300):
            if render:
                try:
                    env.render()
                except Exception:
                    pass

            if not use_random:
                action, current_agent = get_action_from_agent(current_agent, obs)
                if action is None:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            # step
            try:
                step_res = env.step(action)
            except Exception:
                # older gym might return (obs, reward, done, info)
                step_res = env.step(action)
            # normalize step_res to (obs, reward, done, info)
            if isinstance(step_res, tuple) and len(step_res) == 4:
                obs, reward, done, info = step_res
            elif isinstance(step_res, tuple) and len(step_res) == 5:
                # new gym returns (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                # try unpacking simply
                try:
                    obs, reward, done, info = step_res
                except Exception:
                    # unexpected format
                    raise RuntimeError(f"Unexpected env.step return format: {step_res}")

            # print(f"Step {steps}: action={action}, obs={obs}, reward={reward}, done={done}, info={info}")

            # append trajectory (assume obs[0], obs[1] are x,y)
            try:
                traj.append((float(obs[0]), float(obs[1])))
            except Exception:
                if isinstance(obs, dict) and 'position' in obs:
                    p = obs['position']
                    traj.append((float(p[0]), float(p[1])))
                else:
                    # no coordinates available, append previous
                    traj.append(traj[-1])

            # accumulate reward
            try:
                ep_return += float(reward)
            except Exception:
                try:
                    ep_return += float(np.sum(reward))
                except Exception:
                    pass

            # accumulate cost heuristically from info
            step_cost = info['violation']
            if step_cost is None:
                step_cost = 0.0
            try:
                ep_cost += float(step_cost)
            except Exception:
                pass

            if ep_cost > 0:
                any_violation = True

            steps += 1

        print("Are there any violations in this episode?", any_violation)
        all_trajectories.append(np.array(traj))
        elapsed = time.time() - start_time
        per_episode_metadata.append({'steps': steps, 'return': ep_return, 'costs': ep_cost})
        returns.append(ep_return)
        costs.append(ep_cost)
        comp_time.append(elapsed)
        if any_violation:
            episodes_with_violation += 1

        # optionally save per-episode plot
        if per_episode_dir is not None:
            per_episode_dir = Path(per_episode_dir)
            per_episode_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6,6))
            draw_env_map(ax, env)
            xy = all_trajectories[-1]
            ax.plot(xy[:,0], xy[:,1], '-', linewidth=2)
            ax.scatter(xy[0,0], xy[0,1], marker='o', label='start')
            ax.scatter(xy[-1,0], xy[-1,1], marker='x', label='end')
            ax.legend()
            fig.savefig(per_episode_dir / f'episode_{ep+1:03d}.png', dpi=150)
            plt.close(fig)

    # final overlay plot
    fig, ax = plt.subplots(figsize=(8,8))
    draw_env_map(ax, env)

    colors = plt.cm.get_cmap('tab10')
    for i, xy in enumerate(all_trajectories):

        # Save each trajectory to a csv file
        # print(f"Saving trajectory {i} to CSV")
        # np.savetxt(f"ours_trajectory_{i}.csv", xy, delimiter=",")

        c = colors(i % 10)
        ax.plot(xy[:,0], xy[:,1], '-', linewidth=2, alpha=0.9, color=c, label=f'ep{i+1}')
        ax.scatter(xy[0,0], xy[0,1], marker='o', color=c)
        ax.scatter(xy[-1,0], xy[-1,1], marker='x', color=c)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved overlay trajectory plot to: {out}')
    plt.close(fig)

    # metrics summary
    returns_arr = np.array(returns, dtype=float) if returns else np.array([], dtype=float)

    costs_arr = np.array(costs, dtype=float) if costs else np.array([], dtype=float)

    comp_time_arr = np.array(comp_time, dtype=float) if comp_time else np.array([], dtype=float)

    if returns_arr.size > 0:
        print(f"Return: mean={returns_arr.mean():.6f}, std={returns_arr.std(ddof=0):.6f}")
    else:
        print("Return: no episodes run to compute stats.")

    if costs_arr.size > 0:
        print(f"Cost: mean={costs_arr.mean():.6f}, std={costs_arr.std(ddof=0):.6f}")
    else:
        print("Cost: no episodes run to compute stats.")

    if comp_time_arr.size > 0:
        print(f"Computation time per episode: mean={comp_time_arr.mean():.6f}s, std={comp_time_arr.std(ddof=0):.6f}s")
    else:
        print("Computation time: no episodes run to compute stats.")

    print(f"Episodes with at least one violation (cost>0): {episodes_with_violation} / {episodes}")

    # optionally save metadata summary next to output
    try:
        meta_out = Path(out).with_suffix('.summary.json')
        summary = {
            'episodes': episodes,
            'seed': seed,
            'returns_mean': float(returns_arr.mean()) if returns_arr.size>0 else None,
            'returns_std': float(returns_arr.std(ddof=0)) if returns_arr.size>0 else None,
            'costs_mean': float(costs_arr.mean()) if costs_arr.size>0 else None,
            'costs_std': float(costs_arr.std(ddof=0)) if costs_arr.size>0 else None,
            'episodes_with_violation': int(episodes_with_violation),
            'per_episode': per_episode_metadata
        }
        with open(meta_out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary JSON to: {meta_out}")
    except Exception as e:
        print(f"Failed to write summary JSON: {e}")


# --- CLI -----------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results dir that contains config.json and modelN.pickle (optional)')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help="Checkpoint name or 'latest' (default). If name is given, tries that filename first, then model{n}.pickle.")
    parser.add_argument('--episodes', type=int, default=2, help='Number of episodes to roll out')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    parser.add_argument('--out', type=str, default='all_trajectories.png', help='Path to save overlay plot')
    parser.add_argument('--per_episode_dir', type=str, default=None, help='Directory to save per-episode PNGs')
    parser.add_argument('--render', action='store_true', help='Render env during rollout (not displayed in headless mode)')
    parser.add_argument('--init_states_csv', type=str, default="Traj_points.csv", help='Optional CSV with initial states (x,y) in first two columns')

    args = parser.parse_args()
    run_and_plot(args.results_dir, args.checkpoint, args.episodes, args.seed, args.out, args.per_episode_dir, args.render, args.init_states_csv)
