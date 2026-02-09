"""
Visualization of optimal z (max z such that Vz >= 0) for the boat environment.
Uses linear search over z range to find the maximum z value where Vz is non-negative.
"""
import os
import sys
sys.path.append('.')
from absl import app, flags
import re
import json
import numpy as np
from ml_collections import config_flags, ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colors
import jax
import jax.numpy as jnp
from env.boat_robot import BoatRobot
from jaxrl5.agents import DiffCBF


FLAGS = flags.FLAGS
flags.DEFINE_string('model_location', '', 'model location for boat robot model')


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


label_size = 18
legend_size = 30
ticks_size = 18
location = -0.3
width = 0.5

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': label_size}


def find_max_z_for_positive_vz(agent, observations, z_range=(-49.3, 0.0), num_z=200):
    """
    Find the maximum z such that Vz(obs, z) >= 0 using linear search.
    
    Args:
        agent: The trained agent with vz network
        observations: Batch of observations (batch_size, obs_dim)
        z_range: Tuple of (z_min, z_max) for search range
        num_z: Number of z values to search
    
    Returns:
        z_opt: Array of optimal z values (batch_size,)
               Returns -75 if Vz < 0 for all z values
    """
    batch_size = observations.shape[0]
    
    # Create z list from high to low (so we find max z first)
    z_list = jnp.linspace(z_range[1], z_range[0], num_z)  # [0, -49.30] high to low
    
    # Compute Vz for all (obs, z) pairs
    # observations: (batch_size, obs_dim)
    # We need to evaluate Vz(obs_i, z_j) for all i, j
    
    # Expand observations: (batch_size, 1, obs_dim) -> broadcast with z
    obs_expanded = jnp.expand_dims(observations, axis=1)  # (batch_size, 1, obs_dim)
    obs_tiled = jnp.tile(obs_expanded, (1, num_z, 1))  # (batch_size, num_z, obs_dim)
    obs_flat = obs_tiled.reshape(-1, observations.shape[-1])  # (batch_size * num_z, obs_dim)
    
    # Expand z: (num_z,) -> (batch_size, num_z, 1)
    z_expanded = jnp.tile(z_list[None, :, None], (batch_size, 1, 1))  # (batch_size, num_z, 1)
    z_flat = z_expanded.reshape(-1, 1)  # (batch_size * num_z, 1)
    
    # Compute Vz for all pairs
    vz_flat = agent.vz.apply_fn(
        {"params": agent.vz.params},
        obs_flat,
        z_flat
    )
    
    # Reshape back to (batch_size, num_z)
    vz_matrix = vz_flat.reshape(batch_size, num_z)
    
    # Find first z (from high to low) where Vz >= 0
    cond = vz_matrix >= 0.1  # (batch_size, num_z)
    found_any = jnp.any(cond, axis=1)  # (batch_size,)
    first_idx = jnp.argmax(cond, axis=1)  # Index of first True (or 0 if none)
    
    # Select z or sentinel (-75)
    z_opt = jnp.where(found_any, z_list[first_idx], -75.0)
    
    return np.asarray(z_opt), np.asarray(vz_matrix)


def plot_zopt_map(ax, agent, z_range=(-49.3, 0.0), num_z=200, vmin=-75, vmax=0, cb=False):
    """Plot the optimal z map (max z such that Vz >= 0)."""
    # Generate batch observations (grid over x1, x2)
    x1 = np.linspace(-3.0, 2.0, 101)
    x2 = np.linspace(-2.0, 2.0, 101)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    
    batch_size = len(flatten_x1)
    batch_obses = np.zeros((batch_size, 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    # Find optimal z for each observation
    z_opt, _ = find_max_z_for_positive_vz(
        agent, 
        jax.device_put(batch_obses), 
        z_range=z_range, 
        num_z=num_z
    )
    
    z_opt_square = z_opt.reshape(x1_grid.shape)
    
    # Plot contour
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    ct = ax.contourf(
        x1_grid, x2_grid, z_opt_square,
        norm=norm,
        levels=30,
        cmap='rainbow',
    )
    
    # Add contour lines at specific z values
    contour_levels = [-400, -300, -200, -100]
    valid_levels = [l for l in contour_levels if vmin <= l <= vmax]
    if valid_levels:
        ct_line = ax.contour(
            x1_grid, x2_grid, z_opt_square,
            levels=valid_levels, colors='white',
            linewidths=1.0, linestyles='solid', alpha=0.7
        )
        ax.clabel(ct_line, inline=True, fontsize=10, fmt='%.0f')
    
    # Mark infeasible region (where z_opt = -75)
    infeasible_contour = ax.contour(
        x1_grid, x2_grid, z_opt_square,
        levels=[-499], colors='black',
        linewidths=2.0, linestyles='dashed'
    )
    
    if cb:
        cb_obj = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02)
        cb_obj.ax.tick_params(labelsize=ticks_size)
        cb_obj.set_label('Optimal z', fontsize=label_size)
    
    return ax, z_opt_square


def plot_zopt_visualization(env, agent, model_location, z_range=(-49.3, 0.0), num_z=200):
    """Create visualization of optimal z map."""
    
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 4),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3, 3.01, 1.5)
    my_y_ticks = np.arange(-3, 3.01, 1.5)
    
    # Plot 1: Optimal z map
    ax1, z_opt_square = plot_zopt_map(axes[0], agent, z_range=z_range, num_z=num_z, 
                                       vmin=-75, vmax=0, cb=True)
    ax1.set_title(f'Max z s.t. Vz(s,z)≥0\n(z∈[{z_range[0]:.1f}, {z_range[1]:.1f}])', 
                  fontsize=label_size)
    
    # Plot 2: Vc (safe value) for comparison
    x1 = np.linspace(-3.0, 2.0, 101)
    x2 = np.linspace(-2.0, 2.0, 101)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    vc_value = agent.safe_value.apply_fn(
        {"params": agent.safe_value.params},
        jax.device_put(batch_obses)
    )
    vc_flatten = np.asarray(vc_value)
    vc_square = vc_flatten.reshape(x1_grid.shape)
    
    vc_vmin = np.percentile(vc_flatten, 2)
    vc_vmax = np.percentile(vc_flatten, 98)
    norm_vc = colors.Normalize(vmin=vc_vmin, vmax=vc_vmax)
    
    ct = axes[1].contourf(x1_grid, x2_grid, vc_square, norm=norm_vc, levels=30, cmap='rainbow')
    ct_line = axes[1].contour(x1_grid, x2_grid, vc_square, levels=[0], colors='#32ABD6',
                               linewidths=2.0, linestyles='solid')
    axes[1].clabel(ct_line, inline=True, fontsize=15, fmt=r'0')
    axes[1].set_title('Vc (Safe Value)', fontsize=label_size)
    cb_vc = plt.colorbar(ct, ax=axes[1], shrink=0.8, pad=0.02)
    cb_vc.ax.tick_params(labelsize=ticks_size)
    
    # Format all axes
    for ax in axes:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-2.7, 1.7))
        ax.set_ylim((-1.7, 1.7))
        ax.tick_params(labelsize=ticks_size)
        ax.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
    
    # Save figure
    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")
    
    plt.savefig(f"{model_location}/imgs/viz_zopt.png", dpi=600, bbox_inches='tight')
    print(f"Saved figure to {model_location}/imgs/viz_zopt.png")
    
    # Print statistics
    print(f"\nOptimal z statistics:")
    print(f"  Min z_opt: {z_opt_square.min():.2f}")
    print(f"  Max z_opt: {z_opt_square.max():.2f}")
    print(f"  Mean z_opt: {z_opt_square.mean():.2f}")
    feasible_mask = z_opt_square > -75
    print(f"  Feasible points: {feasible_mask.sum()} / {z_opt_square.size} ({100*feasible_mask.mean():.1f}%)")
    if feasible_mask.sum() > 0:
        print(f"  Mean z_opt (feasible only): {z_opt_square[feasible_mask].mean():.2f}")


def plot_zopt_vs_analytical(env, agent, model_location, z_range=(-49.3, 0.0), num_z=200):
    """Compare learned z_opt with analytical z_opt from opt_value_func_calc_jax_batch."""
    
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(12, 4),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3, 3.01, 1.5)
    my_y_ticks = np.arange(-3, 3.01, 1.5)
    
    # Generate observations
    x1 = np.linspace(-3.0, 2.0, 101)
    x2 = np.linspace(-2.0, 2.0, 101)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    batch_obses_jax = jax.device_put(batch_obses)
    
    # Plot 1: Learned z_opt from Vz
    z_opt_learned, _ = find_max_z_for_positive_vz(agent, batch_obses_jax, z_range=z_range, num_z=num_z)
    z_opt_learned_square = z_opt_learned.reshape(x1_grid.shape)
    
    vmin, vmax = -75, 0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ct1 = axes[0].contourf(x1_grid, x2_grid, z_opt_learned_square, norm=norm, levels=30, cmap='rainbow')
    axes[0].set_title('Learned z* from Vz', fontsize=label_size)
    plt.colorbar(ct1, ax=axes[0], shrink=0.8, pad=0.02)
    
    # Plot 2: Analytical z_opt from V and Vc
    v_values = agent.value.apply_fn({"params": agent.value.params}, batch_obses_jax)
    vc_values = agent.safe_value.apply_fn({"params": agent.safe_value.params}, batch_obses_jax)
    
    # Use the analytical formula: V = min(Vh, Vr - z) >= 0
    # z_opt = max z such that min(-Vc, V - z) >= 0
    z_opt_analytical = agent.opt_value_func_calc_jax_batch(
        v_values, -vc_values, (z_range[0] * 100, z_range[1] * 100), num_z
    )
    z_opt_analytical = np.asarray(z_opt_analytical) / 100.0  # Scale back
    z_opt_analytical_square = z_opt_analytical.reshape(x1_grid.shape)
    
    # Clip analytical values for visualization
    z_opt_analytical_square = np.clip(z_opt_analytical_square, -75, 0)
    
    ct2 = axes[1].contourf(x1_grid, x2_grid, z_opt_analytical_square, norm=norm, levels=30, cmap='rainbow')
    axes[1].set_title('Analytical z* from V, Vc', fontsize=label_size)
    plt.colorbar(ct2, ax=axes[1], shrink=0.8, pad=0.02)
    
    # Plot 3: Difference
    diff = z_opt_learned_square - z_opt_analytical_square
    diff_max = max(abs(diff.min()), abs(diff.max()))
    norm_diff = colors.Normalize(vmin=-diff_max, vmax=diff_max)
    ct3 = axes[2].contourf(x1_grid, x2_grid, diff, norm=norm_diff, levels=30, cmap='RdBu_r')
    axes[2].set_title('Difference (Learned - Analytical)', fontsize=label_size)
    plt.colorbar(ct3, ax=axes[2], shrink=0.8, pad=0.02)
    
    # Format all axes
    for ax in axes:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-2.7, 1.7))
        ax.set_ylim((-1.7, 1.7))
        ax.tick_params(labelsize=ticks_size)
        ax.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(width)
            spine.set_color('white')
    
    # Save figure
    plt.savefig(f"{model_location}/imgs/viz_zopt_comparison.png", dpi=600, bbox_inches='tight')
    print(f"Saved figure to {model_location}/imgs/viz_zopt_comparison.png")
    
    # Print comparison statistics
    print(f"\nComparison statistics:")
    print(f"  Learned z_opt range: [{z_opt_learned_square.min():.2f}, {z_opt_learned_square.max():.2f}]")
    print(f"  Analytical z_opt range: [{z_opt_analytical_square.min():.2f}, {z_opt_analytical_square.max():.2f}]")
    print(f"  Difference range: [{diff.min():.2f}, {diff.max():.2f}]")
    print(f"  Mean absolute difference: {np.abs(diff).mean():.2f}")


def load_diffusion_model(model_location):
    """Load the trained model."""
    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    env = BoatRobot(seed=0)

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls")
    agent = globals()[model_cls].create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )

    def get_model_file():
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path

    model_file = get_model_file()
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent


def main(_):
    env, agent = load_diffusion_model(FLAGS.model_location)
    
    # Define z range (should match training range)
    z_range = (-49.3, 0.0)
    num_z = 200
    
    # Plot optimal z visualization
    plot_zopt_visualization(env, agent, FLAGS.model_location, z_range=z_range, num_z=num_z)
    
    # Plot comparison with analytical formula
    plot_zopt_vs_analytical(env, agent, FLAGS.model_location, z_range=z_range, num_z=num_z)


if __name__ == '__main__':
    app.run(main)
