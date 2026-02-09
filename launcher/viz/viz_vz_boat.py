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


def plot_vz_for_z(ax, agent, z_value, vmin=-3.5, vmax=1.01, cb=False):
    """Plot vz value function for a specific z value."""
    # Generate batch observations (grid over x1, x2)
    x1 = np.linspace(-3.0, 2.0, 201)
    x2 = np.linspace(-2.0, 2.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    
    batch_size = len(flatten_x1)
    batch_obses = np.zeros((batch_size, 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    # Create z array with same batch size
    z_array = np.full((batch_size, 1), z_value, dtype=np.float32)
    
    # Compute vz value
    vz_value = agent.vz.apply_fn(
        {"params": agent.vz.params}, 
        jax.device_put(batch_obses), 
        jax.device_put(z_array)
    )
    
    value_flatten = np.asarray(vz_value)
    value_square = value_flatten.reshape(x1_grid.shape)
    
    # Plot contour with explicit levels matching the colorbar range
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 31)
    
    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=levels,
        cmap='rainbow_r',
        extend='both',
    )
    
    # Add zero contour line
    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0], colors='#32ABD6',
        linewidths=2.0, linestyles='solid'
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0')
    
    if cb:
        cb_obj = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(vmin, vmax, 6))
        cb_obj.ax.tick_params(labelsize=ticks_size)
        cbarlabels = cb_obj.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in cbarlabels]
    
    # Add title with z value
    ax.set_title(f'z = {z_value:.0f}', fontsize=label_size, fontname='Times New Roman')
    
    return ax


def plot_vz_multiple_z(env, agent, model_location, z_values=None):
    """Plot vz for multiple z values."""
    
    if z_values is None:
        # Default z values spanning the range [-49.30, 0]
        z_values = [-49.30, -30.000, -10.000, -5.000, 0]
    
    n_plots = len(z_values)
    
    fig, axes = plt.subplots(
        nrows=1, ncols=n_plots,
        figsize=(3.0 * n_plots, 3.0),
        constrained_layout=True,
    )
    
    if n_plots == 1:
        axes = [axes]
    
    my_x_ticks = np.arange(-3, 3.01, 1.5)
    my_y_ticks = np.arange(-3, 3.01, 1.5)
    
    # Compute global min/max for consistent colorbar
    # Sample a few z values to estimate range
    sample_vz_values = []
    x1 = np.linspace(-3.0, 2.0, 51)
    x2 = np.linspace(-2.0, 2.0, 51)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    for z_val in z_values:
        z_array = np.full((len(flatten_x1), 1), z_val, dtype=np.float32)
        vz_value = agent.vz.apply_fn(
            {"params": agent.vz.params},
            jax.device_put(batch_obses),
            jax.device_put(z_array)
        )
        sample_vz_values.append(np.asarray(vz_value))
    
    all_values = np.concatenate(sample_vz_values)
    # Fixed colorbar range
    vmin = -4.5
    vmax = 2.5
    
    print(f"Vz value range: [{all_values.min():.2f}, {all_values.max():.2f}]")
    print(f"Vz plot range (fixed): [{vmin:.2f}, {vmax:.2f}]")
    
    # Plot each z value
    for i, (ax, z_val) in enumerate(zip(axes, z_values)):
        cb = (i == n_plots - 1)  # Add colorbar only to last plot
        ax = plot_vz_for_z(ax, agent, z_val, vmin=vmin, vmax=vmax, cb=cb)
        
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
    
    # Add overall title
    fig.suptitle('Vz Value Function for Different z Values', fontsize=label_size + 2, 
                 fontname='Times New Roman', y=1.05)
    
    # Save figure
    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")
    
    plt.savefig(f"{model_location}/imgs/viz_vz_multiple_z.png", dpi=600, bbox_inches='tight')
    print(f"Saved figure to {model_location}/imgs/viz_vz_multiple_z.png")


def plot_vz_comparison_with_vc(env, agent, model_location, z_values=None):
    """Plot comparison of vz for different z values alongside vc (safe_value)."""
    
    if z_values is None:
        z_values = [-150, -100, -50]
    
    n_z = len(z_values)
    
    fig, axes = plt.subplots(
        nrows=1, ncols=n_z + 1,  # +1 for vc
        figsize=(3.0 * (n_z + 1), 3.0),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3, 3.01, 1.5)
    my_y_ticks = np.arange(-3, 3.01, 1.5)
    
    # Generate batch observations
    x1 = np.linspace(-3.0, 2.0, 201)
    x2 = np.linspace(-2.0, 2.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_size = len(flatten_x1)
    batch_obses = np.zeros((batch_size, 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    # Compute vc (safe_value) for first plot
    vc_value = agent.safe_value.apply_fn(
        {"params": agent.safe_value.params},
        jax.device_put(batch_obses)
    )
    vc_flatten = np.asarray(vc_value)
    vc_square = vc_flatten.reshape(x1_grid.shape)
    
    # Compute all vz values first to determine their range
    vz_squares = []
    for z_val in z_values:
        z_array = np.full((batch_size, 1), z_val, dtype=np.float32)
        vz_value = agent.vz.apply_fn(
            {"params": agent.vz.params},
            jax.device_put(batch_obses),
            jax.device_put(z_array)
        )
        vz_flatten = np.asarray(vz_value)
        vz_square = vz_flatten.reshape(x1_grid.shape)
        vz_squares.append(vz_square)
    
    # Compute dynamic range for vz
    all_vz = np.concatenate([vz.ravel() for vz in vz_squares])
    vz_vmin = np.percentile(all_vz, 2)
    vz_vmax = np.percentile(all_vz, 98)
    print(f"Vz value range: [{all_vz.min():.2f}, {all_vz.max():.2f}]")
    print(f"Vz plot range (2-98 percentile): [{vz_vmin:.2f}, {vz_vmax:.2f}]")
    print(f"Vc value range: [{vc_flatten.min():.2f}, {vc_flatten.max():.2f}]")
    
    # Plot vc with its own normalization
    ax = axes[0]
    vc_vmin = np.percentile(vc_flatten, 2)
    vc_vmax = np.percentile(vc_flatten, 98)
    norm_vc = colors.Normalize(vmin=vc_vmin, vmax=vc_vmax)
    ct = ax.contourf(x1_grid, x2_grid, vc_square, norm=norm_vc, levels=30, cmap='rainbow_r')
    ct_line = ax.contour(x1_grid, x2_grid, vc_square, levels=[0], colors='#32ABD6',
                         linewidths=2.0, linestyles='solid')
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0')
    ax.set_title('Vc (Safe Value)', fontsize=label_size, fontname='Times New Roman')
    cb_vc = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02)
    cb_vc.ax.tick_params(labelsize=ticks_size-2)
    
    # Plot vz for each z value with vz normalization
    norm_vz = colors.Normalize(vmin=vz_vmin, vmax=vz_vmax)
    for i, (z_val, vz_square) in enumerate(zip(z_values, vz_squares)):
        ax = axes[i + 1]
        
        ct = ax.contourf(x1_grid, x2_grid, vz_square, norm=norm_vz, levels=30, cmap='rainbow_r')
        ct_line = ax.contour(x1_grid, x2_grid, vz_square, levels=[0], colors='#32ABD6',
                             linewidths=2.0, linestyles='solid')
        ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0')
        ax.set_title(f'Vz (z={z_val:.0f})', fontsize=label_size, fontname='Times New Roman')
        
        # Add colorbar to last plot
        if i == len(z_values) - 1:
            cb_obj = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02)
            cb_obj.ax.tick_params(labelsize=ticks_size-2)
    
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
    
    plt.savefig(f"{model_location}/imgs/viz_vz_vs_vc.png", dpi=600, bbox_inches='tight')
    print(f"Saved figure to {model_location}/imgs/viz_vz_vs_vc.png")


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
    
    # Plot vz for multiple z values
    z_values = [-49.30, -30.000, -10.000, -5.000, 0]
    plot_vz_multiple_z(env, agent, FLAGS.model_location, z_values=z_values)
    
    # Plot comparison of vz with vc
    z_values_comparison = [-49.30, -30.000, -10.000, -5.000, 0]
    plot_vz_comparison_with_vc(env, agent, FLAGS.model_location, z_values=z_values_comparison)
    
    # Plot vz1 if it exists
    if hasattr(agent, 'vz1') and agent.vz1 is not None:
        plot_vz1_multiple_z(env, agent, FLAGS.model_location, z_values=z_values)


def plot_vz1_for_z(ax, agent, z_value, vmin=-3.5, vmax=1.01, cb=False):
    """Plot vz1 value function for a specific z value."""
    # Generate batch observations (grid over x1, x2)
    x1 = np.linspace(-3.0, 2.0, 201)
    x2 = np.linspace(-2.0, 2.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    
    batch_size = len(flatten_x1)
    batch_obses = np.zeros((batch_size, 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    # Create z array with same batch size
    z_array = np.full((batch_size, 1), z_value, dtype=np.float32)
    
    # Compute vz1 value
    vz1_value = agent.vz1.apply_fn(
        {"params": agent.vz1.params}, 
        jax.device_put(batch_obses), 
        jax.device_put(z_array)
    )
    
    value_flatten = np.asarray(vz1_value)
    value_square = value_flatten.reshape(x1_grid.shape)
    
    # Plot contour with explicit levels matching the colorbar range
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 31)
    
    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=levels,
        cmap='rainbow_r',
        extend='both',
    )
    
    # Add zero contour line
    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0], colors='#32ABD6',
        linewidths=2.0, linestyles='solid'
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0')
    
    if cb:
        cb_obj = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(vmin, vmax, 6))
        cb_obj.ax.tick_params(labelsize=ticks_size)
        cbarlabels = cb_obj.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in cbarlabels]
    
    # Add title with z value
    ax.set_title(f'z = {z_value:.1f}', fontsize=label_size, fontname='Times New Roman')
    
    return ax


def plot_vz1_multiple_z(env, agent, model_location, z_values=None):
    """Plot vz1 for multiple z values. vz1 learns min(V - z, -Vc)."""
    
    if z_values is None:
        z_values = [-49.3, -30.0, -10.0, -5.0, 0]
    
    n_plots = len(z_values)
    
    fig, axes = plt.subplots(
        nrows=1, ncols=n_plots,
        figsize=(3.0 * n_plots, 3.0),
        constrained_layout=True,
    )
    
    if n_plots == 1:
        axes = [axes]
    
    my_x_ticks = np.arange(-3, 3.01, 0.15)
    my_y_ticks = np.arange(-3, 3.01, 0.15)
    
    # Compute global min/max for consistent colorbar
    sample_vz1_values = []
    x1 = np.linspace(-3.0, 2.0, 151)
    x2 = np.linspace(-2.0, 2.0, 151)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 2), dtype=np.float32)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2
    
    for z_val in z_values:
        z_array = np.full((len(flatten_x1), 1), z_val, dtype=np.float32)
        vz1_value = agent.vz1.apply_fn(
            {"params": agent.vz1.params},
            jax.device_put(batch_obses),
            jax.device_put(z_array)
        )
        sample_vz1_values.append(np.asarray(vz1_value))
    
    all_values = np.concatenate(sample_vz1_values)
    # Fixed colorbar range
    vmin = -4.5
    vmax = 2.5
    
    print(f"Vz1 value range: [{all_values.min():.2f}, {all_values.max():.2f}]")
    print(f"Vz1 plot range (fixed): [{vmin:.2f}, {vmax:.2f}]")
    
    # Plot each z value
    for i, (ax, z_val) in enumerate(zip(axes, z_values)):
        cb = (i == n_plots - 1)
        ax = plot_vz1_for_z(ax, agent, z_val, vmin=vmin, vmax=vmax, cb=cb)
        
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
    
    # Add overall title
    fig.suptitle('Vz1 (learns min(V-z, -Vc)) for Different z Values', fontsize=label_size + 2, 
                 fontname='Times New Roman', y=1.05)
    
    # Save figure
    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")
    
    plt.savefig(f"{model_location}/imgs/viz_vz1_multiple_z.png", dpi=600, bbox_inches='tight')
    print(f"Saved figure to {model_location}/imgs/viz_vz1_multiple_z.png")


if __name__ == '__main__':
    app.run(main)
