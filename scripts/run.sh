#!/usr/bin/env bash
# Train EpiFlow on Safe Boat Navigation (env_id 30), evaluate it on the 500 fixed
# initial states used in the paper and draw the value-function figures.
# Run from the repository root.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=False

SEED=${SEED:-0}
python launcher/examples/train_offline.py --env_id 30 --config configs/train_config.py:epiflow \
    --config.seed=${SEED} --experiment_name EpiFlow

RUN=$(ls -td results/BoatRobot/EpiFlow_*_s${SEED}_* | head -1)
python launcher/examples/evaluate_policy.py --run_dir ${RUN}
python launcher/viz/viz_boat.py --run_dir ${RUN}
