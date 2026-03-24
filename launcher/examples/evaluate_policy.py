import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=INFO, 2=WARNING, 3=ERROR -> 3 hides warnings
# optionally reduce XLA autotuner verbosity / behavior (jaxlib version dependent)
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=1'

# also silence the JAX xla_bridge logger once Python logging is available
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


import argparse
import json
import sys
from pathlib import Path
import random

# ensure repository root is on sys.path when running from the results dir
sys.path.append('.')

import gymnasium as gym
from jaxrl5.wrappers import wrap_gym
import jaxrl5.agents as agents_module
from jaxrl5.evaluation import evaluate, evaluate_pr
from env.boat_robot import BoatRobot


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def load_config(results_dir: Path):
    cfg_path = results_dir / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in results_dir: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    return cfg


def construct_env(cfg):
    env_name = cfg.get('env_name')
    if env_name == 'PointRobot':
        # PointRobot is a local custom environment - keep the import flexible
        from env.point_robot import PointRobot
        env = PointRobot(id=0, seed=cfg.get('seed', 0))
        env_max_steps = env._max_episode_steps
        is_pointrobot = True
    elif env_name == 'BoatRobot':
        env = BoatRobot(id=0, seed=cfg.get('seed', 0))
        env_max_steps = env._max_episode_steps
        is_pointrobot = True  # uses evaluate_pr like PointRobot
    else:
        env = gym.make(env_name)
        env_max_steps = env._max_episode_steps
        # wrap with cost_limit from agent kwargs if present
        cost_limit = cfg.get('agent_kwargs', {}).get('cost_limit', None)
        env = wrap_gym(env, cost_limit=cost_limit)
        is_pointrobot = False

    return env, env_max_steps, is_pointrobot


def build_agent_from_config(cfg, seed, obs_space, act_space, env_max_steps):
    agent_kwargs = dict(cfg.get('agent_kwargs', {}))
    # Make sure env_max_steps is passed if needed by the agent
    agent_kwargs['env_max_steps'] = env_max_steps

    # model_cls stored in agent_kwargs during training
    model_cls = agent_kwargs.pop('model_cls', None)
    if model_cls is None:
        raise KeyError("model_cls not found in agent_kwargs of config.json")

    # get the agent class object from jaxrl5.agents
    if hasattr(agents_module, model_cls):
        agent_cls = getattr(agents_module, model_cls)
    else:
        # fallback: try to find in globals of train script style
        try:
            agent_cls = globals()[model_cls]
        except Exception:
            raise ImportError(f"Agent model_cls '{model_cls}' not found in jaxrl5.agents module")

    # Many agents provide a `.create(seed, obs_space, act_space, **kwargs)` factory (seen in training script)
    if hasattr(agent_cls, 'create'):
        agent = agent_cls.create(seed, obs_space, act_space, **agent_kwargs)
    else:
        # fallback to constructing the class directly
        agent = agent_cls(seed, obs_space, act_space, **agent_kwargs)

    return agent


def try_load_checkpoint(agent, ckpt_dir: Path, checkpoint_idx: str):
    def latest(paths):
        return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

    if checkpoint_idx == 'latest':
        model_candidates = sorted(ckpt_dir.glob('model*.pickle'))
        if model_candidates:
            ckpt_path = latest(model_candidates)
        else:
            candidates = list(ckpt_dir.glob('*'))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
            ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        idx_str = str(checkpoint_idx)
        ckpt_path = ckpt_dir / f"model{idx_str}.pickle"
        if ckpt_path is None:
            model_candidates = sorted(ckpt_dir.glob('model*.pickle'))
            ckpt_path = latest(model_candidates) if model_candidates else (ckpt_dir / idx_str)

    print(f"Attempting to load checkpoint from: {ckpt_path}")

    result = agent.load(str(ckpt_path))
    # If load returned a new agent object, return it
    if result is not None:
        print("Loaded checkpoint: agent.load(path) returned a new agent object")

    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the experiment results directory (contains config.json and checkpoints)')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help="Checkpoint to load. Either integer save index (e.g. 1), a filename, or 'latest'.")
    parser.add_argument('--episodes', type=int, default=500, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Enable env.render() during evaluation')
    parser.add_argument('--seed', type=int, default=None, help='Seed to use for evaluation')
    parser.add_argument('--out', type=str, default=None, help='Path to save evaluation results (json)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    cfg = load_config(results_dir)

    # allow CLI to override seed
    seed = args.seed if args.seed is not None else cfg.get('seed', 0)
    random.seed(seed)

    env, env_max_steps, is_pointrobot = construct_env(cfg)

    # build the agent
    agent = build_agent_from_config(cfg, seed, env.observation_space, env.action_space, env_max_steps)

    # try loading the checkpoint
    ckpt_path = Path(results_dir)
    agent = try_load_checkpoint(agent, ckpt_path, args.checkpoint)

    # choose evaluation function
    if is_pointrobot:
        eval_info = evaluate_pr(agent, env, args.episodes)
    else:
        eval_info = evaluate(agent, env, args.episodes, render=args.render)
        # if env supports get_normalized_score, compute it (as training script does)
        if hasattr(env, 'get_normalized_score') and 'return' in eval_info and 'cost' in eval_info:
            try:
                eval_info['normalized_return'], eval_info['normalized_cost'] = env.get_normalized_score(eval_info['return'], eval_info['cost'])
            except Exception as e:
                print(f"get_normalized_score failed: {e}")

    print("Evaluation result:")
    print(eval_info)

    # save results
    out_path = args.out if args.out is not None else str(results_dir / f"eval_result_{args.checkpoint}.json")
    with open(out_path, 'w') as f:
        json.dump(eval_info, f, indent=4)

    print(f"Saved evaluation results to: {out_path}")


if __name__ == '__main__':
    main()
