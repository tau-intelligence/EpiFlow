from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='EpiFlow',
        seed=-1,
        max_steps=100001,
        eval_episodes=1,
        batch_size=2048, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=25000,
        eval_interval=25000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,                                  # DSRL tasks only: l(x) = -25 if unsafe, +1 if safe. Boat: l(x) = signed distance (h)
        pr_data='data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
        br_data='data/boat-1M_mix.hdf5', # The location of boat_robot data
    )

    possible_structures = {
        "epiflow": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="EpiFlow",
                    cost_limit=10,
                    # Actor (FQL flow-based)
                    actor_lr=3e-4,
                    actor_hidden_dims=(512, 512, 512, 512),
                    actor_layer_norm=False,
                    alpha=0.1,            # distillation coefficient
                    flow_steps=10,        # Euler steps for BC flow
                    normalize_q_loss=True,
                    # Critics
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    critic_hyperparam=0.9,       # expectile tau
                    cost_critic_hyperparam=0.9,  # expectile tau
                    critic_objective='expectile',
                    critic_type="hj",
                    value_layer_norm=False,
                    # Epigraph feasibility value function
                    reg_weight=0.25,      # lambda, decomposition regulariser
                    bisection_steps=20,   # binary search for z*(x)
                    z_min=None,           # None -> min(r) / (1 - gamma) over the dataset
                    z_max=None,           # None -> max(r) / (1 - gamma) over the dataset
                    normalize_z=True,
                    # Evaluation
                    N=1,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
