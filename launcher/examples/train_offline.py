import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
from ml_collections import config_flags, ConfigDict
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import EpiFlow
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr, evaluate_br
import json


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_float('alpha', -1.0, 'Override alpha (-1 = use config default)')
flags.DEFINE_float('reg_weight', -1.0, 'Override lambda (-1 = use config default)')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def resolve_z_range(agent_kwargs, ds):
    """Dataset-supported interval [z_min, z_max] for the epigraph variable z."""
    r = ds.dataset_dict['rewards']
    horizon = 1.0 / (1.0 - agent_kwargs.get('discount', 0.99))
    if agent_kwargs.get('z_min') is None:
        agent_kwargs['z_min'] = float(r.min() * horizon)
    if agent_kwargs.get('z_max') is None:
        agent_kwargs['z_max'] = float(r.max() * horizon)
    print(f"Epigraph variable range: z in [{agent_kwargs['z_min']:.3f}, {agent_kwargs['z_max']:.3f}]")


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
    elif details['env_name'] == 'BoatRobot':
        assert details['dataset_kwargs']['br_data'] is not None, "No data for Boat Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['br_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps
    resolve_z_range(config_dict, ds)
    details['agent_kwargs']['z_min'] = config_dict['z_min']
    details['agent_kwargs']['z_max'] = config_dict['z_max']

    out_dir = f"./results/{details['group']}/{details['experiment_name']}"
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(to_dict(details), f, indent=4)

    model_cls = config_dict.pop("model_cls")
    config_dict.pop("cost_scale")
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            print({f"train/{k}": v for k, v in info.items()})

        if i % details['eval_interval'] == 0:
            agent.save(out_dir, save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            elif details['env_name'] == 'BoatRobot':
                eval_info = evaluate_br(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            print({f"eval/{k}": v for k, v in eval_info.items()})

    agent.save(out_dir, "_final")


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']

    if FLAGS.experiment_name == '':
        parameters['experiment_name'] = parameters['agent_kwargs']['model_cls']
    else:
        parameters['experiment_name'] = FLAGS.experiment_name
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_s' + str(parameters['seed']) + '_' + str(random.randint(0,1000))

    if parameters['env_name'] == 'PointRobot' or parameters['env_name'] == 'BoatRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000

    _max_steps_override = os.environ.get('EPIFLOW_MAX_STEPS')
    if _max_steps_override:
        parameters['max_steps'] = int(_max_steps_override)
        print(f"[override] max_steps -> {parameters['max_steps']}")

    if FLAGS.alpha >= 0:
        parameters['agent_kwargs']['alpha'] = FLAGS.alpha
        print(f"[override] alpha -> {FLAGS.alpha}")

    if FLAGS.reg_weight >= 0:
        parameters['agent_kwargs']['reg_weight'] = FLAGS.reg_weight
        print(f"[override] reg_weight -> {FLAGS.reg_weight}")

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")

    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
