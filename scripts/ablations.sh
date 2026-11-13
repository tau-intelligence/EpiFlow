#!/usr/bin/env bash
# Ablations of Section 4.3 / Appendix C. Run from the repository root.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=False
CFG=configs/train_config.py:epiflow
SEEDS=(0 1 2 3 4)

# Regulariser weight lambda (Fig. 4, Fig. 10) on Boat Navigation.
for LAMBDA in 0.0 0.1 0.25 0.5 1.0; do for SEED in "${SEEDS[@]}"; do
  python launcher/examples/train_offline.py --env_id 30 --config ${CFG} --config.seed=${SEED} \
      --reg_weight ${LAMBDA} --experiment_name lambda${LAMBDA}
done; done

# Expectile tau (Fig. 5, Table 3) on Boat Navigation.
for TAU in 0.5 0.7 0.8 0.9; do for SEED in "${SEEDS[@]}"; do
  python launcher/examples/train_offline.py --env_id 30 --config ${CFG} --config.seed=${SEED} \
      --config.agent_kwargs.critic_hyperparam=${TAU} --config.agent_kwargs.cost_critic_hyperparam=${TAU} \
      --experiment_name tau${TAU}
done; done

# Distillation coefficient alpha on Safe Velocity HalfCheetah (Table 6).
for ALPHA in 0.01 0.1 1.0 10.0; do for SEED in "${SEEDS[@]}"; do
  python launcher/examples/train_offline.py --env_id 17 --config ${CFG} --config.seed=${SEED} \
      --alpha ${ALPHA} --experiment_name alpha${ALPHA}
done; done

# Action perturbations (Fig. 9): sigma = 5%, 10%, 20% of the action bound, 100 rollouts.
for ENV in BoatRobot OfflineHalfCheetahVelocityGymnasium-v1 OfflineAntVelocityGymnasium-v1; do
  for SIGMA in 0.05 0.1 0.2; do
    python launcher/examples/evaluate_policy.py --run_dir results/${ENV}/EpiFlow_* \
        --episodes 100 --action_noise ${SIGMA}
  done
done
