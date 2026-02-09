export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu


# python launcher/examples/train_offline.py --env_id 19 --config configs/train_config.py:diffcbf
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf
# python launcher/examples/train_offline.py --env_id 18 --config configs/train_config.py:diffcbf
# python launcher/examples/train_offline.py --env_id 16 --config configs/train_config.py:diffcbf
python launcher/examples/train_offline.py --env_id 20 --config configs/train_config.py:diffcbf

# data quantity
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf --ratio 0.1
# python launcher/examples/train_offline.py --env_id 17 --config configs/train_config.py:diffcbf --ratio 0.5

# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:diffcbf --ratio 0.1
# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:diffcbf --ratio 0.5
