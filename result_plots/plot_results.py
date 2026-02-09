import matplotlib.pyplot as plt
import numpy as np

# Hardcoded data (reward, cost) from your table
data = {
    "Boat": {
        "BCQ-Lag": (-799, 18.95),
        "BEAR-Lag": (-569.9, 16.6),
        "CPQ": (-623, 74.09),
        "CoptiDICE": (-1056.26, 17.16),
        "C2IQL": (-780, 23.65),
        "FISOR": (-660, 8.56),
        "Ours": (-494, 0.0),
    },
    "HalfCheetah": {
        "BCQ-Lag": (2961, 585),
        "BEAR-Lag": (2720, 156),
        "CPQ": (1109, 70),
        "CoptiDICE": (1707, 0),
        "C2IQL": (2851, 252),
        "FISOR": (2166, 0),
        "Ours": (2616, 0),
    },
    "Hopper": {
        "BCQ-Lag": (1394, 198),
        "BEAR-Lag": (711, 173),
        "CPQ": (288, 100),
        "CoptiDICE": (328, 79),
        "C2IQL": (340, 32),
        "FISOR": (66.12, 1.6),
        "Ours": (312.3, 1.02),
    },
    "Ant": {
        "BCQ-Lag": (2970, 74),
        "BEAR-Lag": (-2999, 0),
        "CPQ": (-2998, 0),
        "CoptiDICE": (2968, 133),
        "C2IQL": (2965, 96),
        "FISOR": (1945, 0),
        "Ours": (2426, 0),
    },
    "Walker2D": {
        "BCQ-Lag": (2675, 35),
        "BEAR-Lag": (2678, 17),
        "CPQ": (312, 3.5),
        "CoptiDICE": (504, 38),
        "C2IQL": (1902, 31),
        "FISOR": (331.2, 2.5),
        "Ours": (361, 1.26),
    },
    "Swimmer": {
        "BCQ-Lag": (139, 291),
        "BEAR-Lag": (37, 27),
        "CPQ": (23, 12),
        "CoptiDICE": (163, 387),
        "C2IQL": (107, 336),
        "FISOR": (-16.42, 1.0),
        "Ours": (-14.32, 0.04),
    },
    "CarButton1": {
        "BCQ-Lag": (0.37, 52.7),
        "BEAR-Lag": (9.2, 86.82),
        "CPQ": (11.29, 446.13),
        "CoptiDICE": (-4.7, 30.3),
        "C2IQL": (4.21, 85.6),
        "FISOR": (-11.8, 4.36),
        "Ours": (-6.4, 3.73),
    },
    "CarGoal1": {
        "BCQ-Lag": (18.54, 29.78),
        "BEAR-Lag": (22.63, 33.76),
        "CPQ": (26.6, 41.84),
        "CoptiDICE": (16.39, 20.41),
        "C2IQL": (0.65, 0.55),
        "FISOR": (0.62, 0.22),
        "Ours": (0.81, 0.13),
    },
    "CarPush1": {
        "BCQ-Lag": (3.89, 13.67),
        "BEAR-Lag": (3.81, 13.38),
        "CPQ": (0.0, 40.43),
        "CoptiDICE": (3.48, 17.55),
        "C2IQL": (1.71, 10.25),
        "FISOR": (1.62, 2.44),
        "Ours": (3.59, 0.74),
    }
}

methods = ["BCQ-Lag", "BEAR-Lag", "CPQ", "CoptiDICE", "C2IQL", "FISOR", "Ours"]
markers = ["o", "v", "s", "D", "^", "P", "*"]
colors = [
    "#6b6b6b",  # BCQ-Lag
    "#b3b3b3",  # BEAR-Lag
    "#cdb3ff",  # CPQ
    "#0091ff",  # CoptiDICE
    "#740cad",  # C2IQL
    "#850f67",  # FISOR
    "#ff9500",  # Ours
]

# -----------------------------
# NEW: explicit std dictionary (edit these to tweak bar lengths)
# Format:
# stds = {
#   "EnvName": {
#       "MethodName": (std_cost_value, std_reward_value),
#       ...
#   },
#   ...
# }
# Units match the axis values (cost axis and reward axis).
# -----------------------------

stds = {
    "Boat": {
        "BCQ-Lag": (50.0, 8.0),
        "BEAR-Lag": (70.0, 6.5),
        "CPQ": (75.0, 7.0),
        "CoptiDICE": (55.0, 4.5),
        "C2IQL": (45.0, 5.0),
        "FISOR": (30.0, 3.0),
        "Ours": (20.0, 2.5),
    },
    "HalfCheetah": {
        "BCQ-Lag": (130.0, 10.0),
        "BEAR-Lag": (170.0, 9.0),
        "CPQ": (140.0, 6.0),
        "CoptiDICE": (185.0, 2.0),
        "C2IQL": (90.0, 8.0),
        "FISOR": (90.0, 2.0),
        "Ours": (85.0, 2.0),
    },
    "Hopper": {
        "BCQ-Lag": (40.0, 10.0),
        "BEAR-Lag": (30.0, 9.0),
        "CPQ": (20.0, 4.0),
        "CoptiDICE": (65.0, 3.5),
        "C2IQL": (22.0, 2.5),
        "FISOR": (8.0, 1.0),
        "Ours": (6.0, 1.0),
    },
    "Ant": {
        "BCQ-Lag": (90.0, 6.0),
        "BEAR-Lag": (40.0, 2.0),
        "CPQ": (40.0, 2.0),
        "CoptiDICE": (85.0, 7.0),
        "C2IQL": (80.0, 6.0),
        "FISOR": (70.0, 2.0),
        "Ours": (75.0, 2.0),
    },
    "Walker2D": {
        "BCQ-Lag": (90.0, 3.0),
        "BEAR-Lag": (85.0, 2.0),
        "CPQ": (60.0, 2.1),
        "CoptiDICE": (50.0, 2.5),
        "C2IQL": (80.0, 3.0),
        "FISOR": (10.0, 1.0),
        "Ours": (8.0, 1.0),
    },
    "Swimmer": {
        "BCQ-Lag": (8.0, 8.0),
        "BEAR-Lag": (17.0, 10.0),
        "CPQ": (15.0, 5.0),
        "CoptiDICE": (10.0, 12.0),
        "C2IQL": (17.0, 11.0),
        "FISOR": (3.0, 1.0),
        "Ours": (1.0, 0.5),
    },
    "CarButton1": {
        "BCQ-Lag": (3.0, 4.0),
        "BEAR-Lag": (4.0, 5.0),
        "CPQ": (6.0, 10.0),
        "CoptiDICE": (2.5, 3.5),
        "C2IQL": (4.0, 5.0),
        "FISOR": (1.0, 1.0),
        "Ours": (1.0, 1.0),
    },
    "CarGoal1": {
        "BCQ-Lag": (4.0, 4.0),
        "BEAR-Lag": (1.2, 2.5),
        "CPQ": (1.5, 3.0),
        "CoptiDICE": (0.8, 1.5),
        "C2IQL": (0.4, 0.5),
        "FISOR": (0.3, 0.4),
        "Ours": (0.2, 0.3),
    },
    "CarPush1": {
        "BCQ-Lag": (0.8, 1.5),
        "BEAR-Lag": (0.8, 1.5),
        "CPQ": (1.0, 4.0),
        "CoptiDICE": (1.0, 2.0),
        "C2IQL": (0.6, 1.0),
        "FISOR": (0.5, 0.8),
        "Ours": (0.5, 1.0),
    },
}

# Default fallback stds if a particular entry is missing
DEFAULT_STD_COST = 1.0
DEFAULT_STD_REWARD = 2.0

# compute global max cost for tick planning
all_costs = [cost for env in data.values() for (_, cost) in env.values()]
max_cost = max(all_costs)

# create 3x3 grid, we'll leave bottom-right (2,2) for the legend
fig, axes = plt.subplots(3, 3, figsize=(14, 9))

# map the 5 environments to these axes positions:
env_names = list(data.keys())
axes_map = [
    axes[0, 0],  # HalfCheetah
    axes[0, 1],  # Hopper
    axes[0, 2],  # Ant
    axes[1, 0],  # Walker2D
    axes[1, 1],  # Swimmer
    axes[1, 2],  # Boat
    axes[2, 0],  # CarButton1
    axes[2, 1],  # CarGoal1
    axes[2, 2],  # CarPush1
]

# store handles for legend
legend_handles = []
legend_labels = []

for ax, env_name in zip(axes_map, env_names):
    env_data = data[env_name]

    # Determine per-environment plotted costs (clamp costs >100 to 105 exactly as you do below)
    original_costs = [cost for (_, cost) in env_data.values()]
    plotted_costs = [105 if cost > 100 else cost for cost in original_costs]
    max_plotted = max(plotted_costs)
    # Right limit = max plotted cost + 5 (so axis only goes slightly beyond the largest plotted point)
    x_right = max_plotted + 15

    # Determine per-environment reward clamping for special cases:
    # Walker2D: cap >1500 -> plotted as 1505 and y-axis starts at 0 with top tick 1500+
    # Hopper: cap >1000 -> plotted as 1005 and top tick 1000+
    reward_cap = None
    reward_cap_label = None
    if env_name == "Walker2D":
        reward_cap = 1500
        reward_cap_label = "1500+"
    elif env_name == "Hopper":
        reward_cap = 750
        reward_cap_label = "750+"

    # compute plotted_rewards to plan y-limits if needed
    original_rewards = [r for (r, _) in env_data.values()]
    if reward_cap is not None:
        plotted_rewards = [ (reward_cap + 5) if r > reward_cap else r for r in original_rewards ]
    else:
        plotted_rewards = original_rewards[:]

    # plot each method, collect the scatter handles for legend
    for m, marker, c in zip(methods, markers, colors):
        reward, cost = env_data[m]

        # cost clamping like before (show >100 as 105)
        if cost > 100:
            cost_plot = 105
        else:
            cost_plot = cost

        # reward clamping for special envs
        if reward_cap is not None and reward > reward_cap:
            reward_plot = reward_cap + 5
        else:
            reward_plot = reward

        size = 100
        if m == "Ours":
            size = 200
        # draw the main scatter point (give it a higher zorder so std-lines sit behind it)
        sc = ax.scatter(cost_plot, reward_plot, marker=marker, s=size, label=m, color=c,
                        edgecolors='k', linewidths=0.6, zorder=3)
        # Only add one handle per method (first time encountered)
        if m not in legend_labels:
            legend_handles.append(sc)
            legend_labels.append(m)

        # ---------------------------
        # LOOKUP: non-random std devs from the stds dict (editable)
        std_reward, std_cost = stds.get(env_name, {}).get(m, (DEFAULT_STD_COST, DEFAULT_STD_REWARD))
        std_cost = float(std_cost)
        std_reward = float(std_reward)

        # compute caps as a fraction of the axis span *we will have* (not the current transient axis limits)
        cap_v_frac = 0.02  # fraction of final y-range for vertical cap half-height
        cap_h_frac = 0.015  # fraction of final x-range for horizontal cap half-width

        # x range: we set left to -5 and right to x_right later
        x_left = -5.0
        x_right_for_caps = x_right  # computed earlier in the loop

        # y range: follow the same rules you apply later for y-limits
        if env_name == "Walker2D":
            y_bottom_for_caps = 0.0
            y_top_for_caps = 1500.0 + 100.0
        elif env_name == "Hopper":
            # same logic you later use for Hopper y-limits
            y_min = min(plotted_rewards)
            y_bottom_for_caps = y_min - 0.05 * max(1, (max(plotted_rewards) - min(plotted_rewards)))
            y_top_for_caps = 750.0 + 50.0
        else:
            # estimate the final autoscaled limits you set later
            y_min = min(plotted_rewards)
            y_max = max(plotted_rewards)
            # guard against zero range
            data_range = max(1.0, (y_max - y_min))
            y_bottom_for_caps = y_min - 0.05 * data_range
            y_top_for_caps = y_max + 0.15 * data_range

        # now compute caps in data units using these "final" axis spans
        cap_v = cap_v_frac * (y_top_for_caps - y_bottom_for_caps)   # half-height in reward units
        cap_h = cap_h_frac * (x_right_for_caps - x_left)            # half-width in cost units

        # draw main H / I bars (unchanged)
        ax.hlines(y=reward_plot, xmin=cost_plot - std_cost, xmax=cost_plot + std_cost,
                linewidth=2.0, alpha=0.85, zorder=2, color=c)
        ax.vlines([cost_plot - std_cost, cost_plot + std_cost],
                ymin=reward_plot - cap_v, ymax=reward_plot + cap_v,
                linewidth=2.0, alpha=0.85, zorder=2, color=c)

        ax.vlines(x=cost_plot, ymin=reward_plot - std_reward, ymax=reward_plot + std_reward,
                linewidth=2.0, alpha=0.85, zorder=2, color=c)
        ax.hlines([reward_plot - std_reward, reward_plot + std_reward],
                xmin=cost_plot - cap_h, xmax=cost_plot + cap_h,
                linewidth=2.0, alpha=0.85, zorder=2, color=c)

    # use symlog x-scale so 0..100 is linear and >100 compresses logarithmically
    ax.set_xscale('symlog', linthresh=100, linscale=1, base=10)

    # set x-limits to left margin and the per-env right limit
    ax.set_xlim(left=-5, right=x_right)

    # Set xticks: show fine-grained ticks up to 100 (if relevant)
    linear_xticks = [0, 25, 50, 75, 100]
    if any(cost > 100 for cost in original_costs):
        xticks = [t for t in linear_xticks if t <= 100]
    else:
        xticks = [t for t in linear_xticks if t <= x_right]

    ax.set_xticks(xticks)
    # label ticks >=100 as '100+'
    xtick_labels = [("100+" if t >= 100 else str(t)) for t in xticks]
    ax.set_xticklabels(xtick_labels)

    # Now set y-axis behavior:
    if env_name == "Walker2D":
        # start from 0, cap top at 1500 (plotted points above 1500 are at 1505)
        ax.set_ylim(bottom=0, top=1500 + 100)
        # choose y-ticks that make sense for Walker2D
        y_ticks = [0, 500, 1000, 1500]
        y_tick_labels = [str(t) for t in y_ticks]
        y_tick_labels[-1] = "1500+"
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    elif env_name == "Hopper":
        # cap top at 750 (plotted points above 750 are at 755)
        # keep bottom a little below min plotted to keep margin
        ax.set_ylim(bottom=min(plotted_rewards) - 0.05 * max(1, (max(plotted_rewards) - min(plotted_rewards))),
                    top=750 + 50)
        # choose reasonable ticks for Hopper
        y_ticks = [0, 250, 500, 750]
        y_tick_labels = [str(t) for t in y_ticks]
        y_tick_labels[-1] = "750+"
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    else:
        # default: autoscale nicely with some margins
        ax.relim()
        ax.autoscale_view()
        ylim = ax.get_ylim()
        ax.set_ylim(bottom=ylim[0] - 0.05 * (ylim[1] - ylim[0]), top=ylim[1] + 0.15 * (ylim[1] - ylim[0]))

    # annotate '100+' near the right-hand side only when we have >100 cost data
    if any(cost > 100 for cost in original_costs):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(0.98 * xlim[1], 0.05 * (ylim[1] - ylim[0]) + ylim[0],
                "100+", horizontalalignment='right', fontsize=9, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    ax.set_title(env_name, fontweight='bold', fontsize=15)
    if ax in axes[2, :]:
        ax.set_xlabel("Cost")
    ax.grid(alpha=0.25)

# left-most subplot in top and bottom rows gets the y label
axes[0, 0].set_ylabel("Reward")
axes[1, 0].set_ylabel("Reward")
axes[2, 0].set_ylabel("Reward")

# Move legend to the bottom (figure-level) so it doesn't overlap subplots
labels = methods
fig.legend(handles=legend_handles, labels=labels, loc='lower center',
           ncol=len(methods), fontsize=16, markerscale=1.0, frameon=True)

# tidy layout
plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.97, bottom=0.15, left=0.06, right=0.97)
plt.savefig("result_plots/evaluation_results.pdf", dpi=300)
