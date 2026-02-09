"""
load_and_evaluate_model.py

Usage examples:
  python load_and_evaluate_model.py --results_dir ./results/PointRobot/my_experiment --checkpoint latest --eval_episodes 20 --render True
  python load_and_evaluate_model.py --results_dir ./results/PointRobot/my_experiment --checkpoint 3

This script tries to reconstruct the agent from the saved `config.json` in the experiment folder,
locate a checkpoint (either a specific number or the latest), load it, and run evaluation with render=True.
It mirrors the environment/agent construction in your training script.

Notes:
- The exact mechanics of `agent.load(...)` and how checkpoints are stored can vary across agent implementations.
  This script attempts several common loading strategies and prints helpful messages if loading fails.
- If your agent implementation uses a different checkpoint naming convention, point `--checkpoint` to the exact
  checkpoint identifier (integer index or filename) or adjust the `find_latest_candidate()` logic below.
"""

import os
import sys
import json
import argparse
import datetime
import traceback

# allow imports from project root
sys.path.append('.')

import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
import jaxrl5.agents as agents_mod
from jaxrl5.evaluation import evaluate, evaluate_pr


def find_latest_candidate(results_dir):
    """Try to find a likely latest checkpoint identifier or file in results_dir.
    Returns either an integer-like token or a path. If nothing obvious found, returns None.
    """
    # look for integer-named entries (common when you save with incremental integer ids)
    entries = os.listdir(results_dir)
    ints = []
    for e in entries:
        name = os.path.splitext(e)[0]
        if name.isdigit():
            try:
                ints.append(int(name))
            except Exception:
                pass
    if ints:
        return max(ints)

    # fallback: choose most recently modified file/dir in results_dir (excluding config.json)
    candidates = [os.path.join(results_dir, e) for e in entries if e != 'config.json']
    if not candidates:
        return None
    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    # return the path (full path) of the latest candidate
    return candidates[0]


def try_agent_load(agent, results_dir, checkpoint_token=None):
    """Try several common load call forms for an agent.
    Returns True on success.
    """
    tried = []
    # 1) If agent has a load method that accepts (path, token)
    results_dir = results_dir + '/model5.pickle'
    try:
        if checkpoint_token is None:
            agent.load(results_dir)
        else:
            # try integer and string forms
            agent.load(results_dir, checkpoint_token)
        return True
    except Exception as e:
        tried.append(f"agent.load(results_dir, token) failed: {e}")

    # 2) try loading by path to a subfile/dir
    if checkpoint_token is not None:
        candidate_paths = [
            os.path.join(results_dir, str(checkpoint_token)),
            os.path.join(results_dir, f"checkpoint_{checkpoint_token}"),
            os.path.join(results_dir, f"ckpt_{checkpoint_token}"),
            os.path.join(results_dir, f"{checkpoint_token}.pth"),
            os.path.join(results_dir, f"{checkpoint_token}.pt"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    agent.load(p)
                    return True
                except Exception as e:
                    tried.append(f"agent.load('{p}') failed: {e}")

    # 3) try agent.load with no args (some implementations do this)
    try:
        agent.load()
        return True
    except Exception as e:
        tried.append(f"agent.load() failed: {e}")

    print("Tried the following load attempts but all failed:")
    for t in tried:
        print(" - ", t)
    return False


def build_agent_from_config(config, env, seed):
    """Reconstruct the agent using config['agent_kwargs'].
    This closely follows the creation logic from your training script.
    """
    agent_kwargs = dict(config.get('agent_kwargs', {}))

    # ensure env_max_steps is present (training script added this before creating agent)
    env_max_steps = config.get('env_max_steps') or getattr(env, '_max_episode_steps', None)
    if env_max_steps is not None:
        agent_kwargs['env_max_steps'] = env_max_steps

    # remove items that agent.create may not expect
    # training code explicitly popped cost_scale before create
    if 'cost_scale' in agent_kwargs:
        agent_kwargs.pop('cost_scale')

    model_cls = agent_kwargs.pop('model_cls', None)
    if model_cls is None:
        raise RuntimeError("model_cls not found in agent_kwargs inside config.json")

    # try to get the class from the agents module
    if hasattr(agents_mod, model_cls):
        model_class = getattr(agents_mod, model_cls)
    else:
        # maybe it was imported into globals in the training env; try global lookup
        model_class = globals().get(model_cls)
        if model_class is None:
            raise RuntimeError(f"Agent class {model_cls} not found in jaxrl5.agents or globals()")

    agent = model_class.create(seed, env.observation_space, env.action_space, **agent_kwargs)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the experiment results folder (contains config.json and saved checkpoints)')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help="Checkpoint to load (integer index, filename, or 'latest'). Default: latest")
    parser.add_argument('--eval_episodes', type=int, default=None,
                        help='Number of evaluation episodes. If not provided, uses config value or 10')
    parser.add_argument('--render', type=lambda x: (str(x).lower() in ['1','true','yes']), default=True,
                        help='Whether to render during evaluation (default: True)')
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # load config.json saved by training script
    cfg_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(cfg_path):
        print(f"config.json not found in {results_dir}. Make sure you pointed to the correct experiment folder.")
        return

    with open(cfg_path, 'r') as f:
        config = json.load(f)

    env_name = config.get('env_name')
    if env_name == 'PointRobot':
        env = PointRobot(id=0, seed=0)
        eval_fn = evaluate_pr
    else:
        env = gym.make(env_name)
        # training code wrapped env for evaluation with cost_limit and normalized returns; mirror that
        cost_limit = config.get('agent_kwargs', {}).get('cost_limit')
        if cost_limit is not None:
            env = wrap_gym(env, cost_limit=cost_limit)
        eval_fn = evaluate

    # construct agent
    seed = config.get('seed', 0)
    try:
        agent = build_agent_from_config(config, env, seed)
    except Exception as e:
        print("Failed to construct agent from config:", e)
        traceback.print_exc()
        return

    # choose checkpoint
    checkpoint_token = args.checkpoint
    if checkpoint_token == 'latest' or checkpoint_token is None:
        checkpoint_token = find_latest_candidate(results_dir)
        print("Auto-selected checkpoint token/path:", checkpoint_token)

    # try to load
    success = try_agent_load(agent, results_dir, checkpoint_token=None)
    if not success:
        print("Unable to load checkpoint. Aborting evaluation.")
        return

    # evaluation episodes: config may include eval_episodes
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else config.get('eval_episodes', 10)

    # run evaluation
    print(f"Running evaluation for {eval_episodes} episodes (render={args.render})")
    if env_name == 'PointRobot':
        eval_info = eval_fn(agent, env, eval_episodes)
    else:
        eval_info = eval_fn(agent, env, eval_episodes, render=args.render)
        # compute normalized scores if method exists on env
        if hasattr(env, 'get_normalized_score') and 'return' in eval_info and 'cost' in eval_info:
            try:
                eval_info['normalized_return'], eval_info['normalized_cost'] = env.get_normalized_score(eval_info['return'], eval_info['cost'])
            except Exception:
                pass

    print("Evaluation result:")
    print(json.dumps(eval_info, indent=4))


if __name__ == '__main__':
    main()
