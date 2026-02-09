# Flow Matching Implementation

## Overview

This codebase has been updated to use **Flow Matching** instead of **DDPM (Denoising Diffusion Probabilistic Models)** for policy learning. Flow matching is a more principled approach to generative modeling that often provides:

- Simpler training objectives
- More stable training dynamics
- Faster sampling at inference time
- Better performance in offline RL settings

## Key Differences

### 1. Training Objective

**DDPM (Old):**
- Predicts noise ε that was added to actions
- Uses a schedule of noise levels (βₜ, αₜ, etc.)
- Loss: `||ε_pred - ε||²`

**Flow Matching (New):**
- Predicts velocity field v(x, t) 
- Uses continuous time t ∈ [0, 1]
- Loss: `||v_pred - (x₁ - x₀)||²`

### 2. Sampling Process

**DDPM (Old):**
- Start from pure noise
- Iteratively denoise using predicted noise
- Requires careful handling of noise schedules

**Flow Matching (New):**
- Start from random Gaussian noise x₀
- Follow the learned velocity field using Euler integration
- Simpler and more interpretable

### 3. Code Changes

#### Network Architecture
- `DDPM` → `ActorVectorField`
- Both use the same underlying MLP/ResNet architectures
- Time encoding remains similar (FourierFeatures)

#### Training
```python
# Old DDPM training
noisy_actions = sqrt(α_t) * actions + sqrt(1-α_t) * noise
loss = ||predict_noise(s, noisy_actions, t) - noise||²

# New Flow Matching training  
x_t = (1-t) * x_0 + t * x_1  # Linear interpolation
velocity = x_1 - x_0
loss = ||predict_velocity(s, x_t, t) - velocity||²
```

#### Sampling
```python
# Old DDPM sampling
x = random_noise()
for t in reversed(range(T)):
    x = denoise_step(x, t)

# New Flow Matching sampling
x = random_noise()
for step in range(num_steps):
    t = step / num_steps
    v = predict_velocity(s, x, t)
    x = x + v * dt  # Euler integration
```

## Configuration

To use flow matching, the config parameter `sampling_method` is no longer needed (or can be set to `"flow_matching"`). The key parameters are:

- `T`: Number of integration steps (default: 5-10)
- `N`: Number of action samples for evaluation
- `actor_architecture`: 'mlp' or 'ln_resnet'
- `clip_sampler`: Whether to clip actions during sampling

## References

This implementation is based on:
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [FQL: Flow Q-Learning](https://github.com/seohongpark/fql)
- Implicit Flow Q-Learning (IFQL) approach

## Migration Notes

If you have existing DDPM-trained models:
- They are **not directly compatible** with the flow matching code
- You'll need to retrain models using the new flow matching objective
- Hyperparameters (T, learning rates, etc.) may need adjustment
- Flow matching often works well with fewer integration steps than DDPM

### Unused Parameters

The following parameters are kept for backward compatibility but are **not used** in flow matching:
- `ddpm_temperature`: Temperature for DDPM sampling (flow matching doesn't use this)
- `alphas`, `alpha_hats`, `betas`: DDPM noise schedules (flow matching uses continuous time)
- `beta_schedule`: Schedule type for DDPM (not needed in flow matching)
- `M`: Repeat last step parameter from DDPM (not applicable)

These can be safely ignored when using flow matching.

## Performance

Flow matching typically:
- Trains faster (simpler objective)
- Samples faster (more efficient integration)
- Achieves comparable or better performance on offline RL tasks
- Works especially well with the feasibility-based weighting scheme in DiffCBF
