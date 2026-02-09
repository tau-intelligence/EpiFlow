"""
Example usage of Flow Matching in DiffCBF

This file demonstrates how to use the updated flow matching implementation.
"""

# Key Changes Summary:

# 1. Network definition now uses ActorVectorField instead of DDPM
# The network signature is the same:
#   - Input: (observations, actions, time)
#   - Output: velocity vector (same shape as actions)

# 2. Training objective changed from noise prediction to velocity prediction
# Old DDPM:
#   noise = random_normal()
#   noisy_action = sqrt(alpha_t) * action + sqrt(1-alpha_t) * noise
#   predicted_noise = network(obs, noisy_action, t)
#   loss = MSE(predicted_noise, noise)
#
# New Flow Matching:
#   x_0 = random_normal()  # source
#   x_1 = action           # target
#   t = random_uniform(0, 1)
#   x_t = (1-t) * x_0 + t * x_1
#   velocity = x_1 - x_0
#   predicted_velocity = network(obs, x_t, t)
#   loss = MSE(predicted_velocity, velocity)

# 3. Sampling uses Euler integration instead of denoising
# Old DDPM:
#   x = random_noise()
#   for timestep in reversed(range(T)):
#       x = denoise_step(x, timestep)  # uses noise schedule
#
# New Flow Matching:
#   x = random_noise()
#   dt = 1.0 / num_steps
#   for step in range(num_steps):
#       t = step * dt
#       v = network(obs, x, t)
#       x = x + v * dt  # Euler step
#   return x

# 4. Configuration
# The sampling_method parameter can be set to "flow_matching" or omitted
# Key parameters:
#   - T: number of integration steps (5-10 typically works well)
#   - N: number of samples to generate for action selection
#   - clip_sampler: whether to clip actions to [-1, 1] during sampling

# Example config update:
config_example = {
    "agent_kwargs": {
        "model_cls": "DiffCBF",
        "T": 5,  # Integration steps (fewer needed than DDPM!)
        "N": 16,  # Number of action samples
        "actor_architecture": "ln_resnet",
        "sampling_method": "flow_matching",  # Optional, defaults to flow matching now
        # ... other params
    }
}

# No changes needed to:
# - Critic networks
# - Value networks  
# - Safe critic/value networks
# - Q-value based action selection (maxq/minqc)
# - Feasibility-based weighting

print("Flow matching is now the default sampling method!")
print("The code is backward compatible - just retrain your models.")
