"""Visualisations for Boat Navigation (Figures 1, 7 and 10 of the paper).

    python launcher/viz/viz_boat.py --run_dir results/BoatRobot/<run>

Writes to ``<run_dir>/imgs/``:
  epigraph_value.png   V_hat(x, z) for several z (Fig. 7)
  z_star.png           state-wise budget z*(x); infeasible states masked (Fig. 10)
  trajectories.png     policy rollouts from the initial states in plots/data/Traj_points.csv
"""
import argparse
import os
import sys

sys.path.append(".")

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from launcher.common import load_run

X_RANGE, Y_RANGE, GRID = (-3.0, 2.0), (-2.0, 2.0), 201


def grid():
    xs = np.linspace(*X_RANGE, GRID)
    ys = np.linspace(*Y_RANGE, GRID)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)


def decorate(ax, env):
    for hx, hy, r in env.hazard_position_list:
        ax.add_patch(Circle((hx, hy), r, fill=True, color="k", alpha=0.6))
    ax.plot(*env.goal_position, marker="*", color="gold", markersize=14, markeredgecolor="k")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_epigraph_value(agent, env, out, z_values):
    xx, yy, obs = grid()
    fig, axes = plt.subplots(1, len(z_values), figsize=(3.2 * len(z_values), 3.0),
                             constrained_layout=True)
    values = [np.asarray(agent.epi_v(jnp.asarray(obs), jnp.full((len(obs), 1), z)))
              for z in z_values]
    vmax = max(np.abs(v).max() for v in values)
    for ax, z, v in zip(axes, z_values, values):
        v = v.reshape(xx.shape)
        cf = ax.contourf(xx, yy, v, levels=31, cmap="rainbow_r", vmin=-vmax, vmax=vmax)
        ax.contour(xx, yy, v, levels=[0.0], colors="k", linewidths=2.0)
        decorate(ax, env)
        ax.set_title(f"z = {z:.0f}")
    fig.colorbar(cf, ax=axes, shrink=0.8)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_z_star(agent, env, out):
    xx, yy, obs = grid()
    obs = jnp.asarray(obs)
    z_star = np.asarray(agent.optimal_z(obs))[:, 0]
    feasible = np.asarray(agent.epi_v(obs, jnp.asarray(z_star)[:, None])) >= 0
    z_plot = np.where(feasible, z_star, np.nan).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.set_facecolor("#5b2a86")  # infeasible states
    cf = ax.contourf(xx, yy, z_plot, levels=31, cmap="coolwarm")
    decorate(ax, env)
    fig.colorbar(cf, ax=ax, label=r"$z^*(x)$")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_trajectories(agent, env, out, init_csv):
    data = np.genfromtxt(init_csv, delimiter=",", skip_header=1)[:, :2]
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for s0 in data:
        obs, path = env.reset(state=np.array(s0)), [s0]
        for _ in range(env._max_episode_steps):
            action, agent = agent.eval_actions(obs)
            obs, _, done, _ = env.step(action)
            path.append(obs.copy())
            if done:
                break
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], lw=1.5)
        ax.plot(*s0, "o", color="k", markersize=4)
    decorate(ax, env)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="_final")
    parser.add_argument("--z_values", type=float, nargs="+", default=None,
                        help="Epigraph levels to plot (default: 0%, 40%, 70%, 90%, 100% of [z_min, z_max]).")
    parser.add_argument("--init_states", default="plots/data/Traj_points.csv")
    args = parser.parse_args()

    config, env, agent = load_run(args.run_dir, args.checkpoint)
    assert config["env_name"] == "BoatRobot", "viz_boat.py only supports BoatRobot runs"
    out_dir = os.path.join(args.run_dir, "imgs")
    os.makedirs(out_dir, exist_ok=True)
    z_values = args.z_values or [agent.z_min + f * (agent.z_max - agent.z_min)
                                 for f in (0.0, 0.4, 0.7, 0.9, 1.0)]
    plot_epigraph_value(agent, env, os.path.join(out_dir, "epigraph_value.png"), z_values)
    plot_z_star(agent, env, os.path.join(out_dir, "z_star.png"))
    plot_trajectories(agent, env, os.path.join(out_dir, "trajectories.png"), args.init_states)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
