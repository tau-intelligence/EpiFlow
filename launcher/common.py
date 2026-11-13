"""Rebuild an environment and a trained agent from a results directory."""
import json
import os
import sys

sys.path.append(".")

import gymnasium as gym

from env.boat_robot import BoatRobot
from env.point_robot import PointRobot
from jaxrl5 import agents
from jaxrl5.wrappers import wrap_gym

CUSTOM_ENVS = {"BoatRobot": BoatRobot, "PointRobot": PointRobot}


def make_env(config: dict):
    name = config["env_name"]
    if name in CUSTOM_ENVS:
        env = CUSTOM_ENVS[name](id=0, seed=0)
        return env, env._max_episode_steps
    import dsrl  # noqa: F401  (registers the offline environments)
    env = gym.make(name)
    max_steps = env._max_episode_steps
    return wrap_gym(env, cost_limit=config["agent_kwargs"]["cost_limit"]), max_steps


def load_run(run_dir: str, checkpoint: str = "_final"):
    """Return (config, env, agent) for ``run_dir/model{checkpoint}.pickle``."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    env, env_max_steps = make_env(config)

    kwargs = dict(config["agent_kwargs"])
    model_cls = kwargs.pop("model_cls")
    kwargs.pop("cost_scale", None)
    kwargs["env_max_steps"] = env_max_steps
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    agent = getattr(agents, model_cls).create(
        config["seed"], env.observation_space, env.action_space, **kwargs)
    agent = agent.load(os.path.join(run_dir, f"model{checkpoint}.pickle"))
    return config, env, agent
