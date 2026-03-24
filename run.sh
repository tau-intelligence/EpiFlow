export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME=cuda


# Safe Navigation Gymnasium environments
# python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:diffcbf    #CarButton1
# python launcher/examples/train_offline.py --env_id 4 --config configs/train_config.py:diffcbf    #CarGoal1
# python launcher/examples/train_offline.py --env_id 6 --config configs/train_config.py:diffcbf    #CarPush1



# Safe Velocity Gymnasium environments
# python launcher/examples/train_offline.py --env_id 16 --config configs/train_config.py:diffcbf    # AntVelocity
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf    # HalfCheetahVelocity
# python launcher/examples/train_offline.py --env_id 18 --config configs/train_config.py:diffcbf    # HopperVelocity
# python launcher/examples/train_offline.py --env_id 19 --config configs/train_config.py:diffcbf    # SwimmerVelocity
# python launcher/examples/train_offline.py --env_id 20 --config configs/train_config.py:diffcbf    # Walker2dVelocity

# data quantity
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf --ratio 0.1
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf --ratio 0.5

# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:diffcbf --ratio 0.1
# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:diffcbf --ratio 0.5


# Safe Boat Navigation training
python launcher/examples/train_offline.py --env_id 30 --config configs/train_config.py:diffcbf  # BoatRobot


# For evaluation of all the environments, use the following command (e.g., for Ant (we use '_final' at the end for final checkpoint)):
# python launcher/examples/evaluate_policy.py --results_dir  results/OfflineAntVelocityGymnasium-v1/flow_matching_feasibility_hj_N8_safemaxq_2026-03-16_s49_36_final