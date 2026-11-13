#!/usr/bin/env bash
# Main results (Tables 1 & 2): every task with 5 training seeds, then evaluation of
# all seeds on 500 fixed initial states. Run from the repository root.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=False

# env_id -> see env/env_list.py
BOAT=(30)
SAFETY_GYM=(17 18 16 20 19 0 4 6)      # HalfCheetah Hopper Ant Walker2d Swimmer CarButton1 CarGoal1 CarPush1
BULLET=(22 28 25 23 26 27 21 24)       # Ant/Ball/Car/Drone-Run, Ant/Ball/Car/Drone-Circle
SEEDS=(0 1 2 3 4)

for ENV_ID in "${BOAT[@]}" "${SAFETY_GYM[@]}" "${BULLET[@]}"; do
  ENV_NAME=$(python -c "from env.env_list import env_list; print(env_list[${ENV_ID}])")
  for SEED in "${SEEDS[@]}"; do
    python launcher/examples/train_offline.py --env_id ${ENV_ID} \
        --config configs/train_config.py:epiflow --config.seed=${SEED} --experiment_name EpiFlow
  done
  python launcher/examples/evaluate_policy.py --run_dir results/${ENV_NAME}/EpiFlow_*
done
