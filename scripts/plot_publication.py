import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import matplotlib as mpl

# Set publication quality style
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42, # TrueType fonts
    'ps.fonttype': 42,
})
plt_sns.set_style("whitegrid")
plt_sns.set_palette("colorblind")

def load_trajectories(log_dir="logs"):
    files = glob.glob(os.path.join(log_dir, "*.json"))
    data = []
    for file in files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    return data

def plot_pareto_front(data, output_dir="plots", fmt="both"):
    os.makedirs(output_dir, exist_ok=True)

    j_damage = []
    j_spatial = []

    for episode in data:
        total_damage = 0.0
        total_spatial = 0.0
        decision_steps = 0

        for step in episode:
            # Check if it has the exact metrics requested in the prompt
            if "J_damage" in step and "J_spatial" in step:
                # To prevent double counting if decisions aren't taken every step,
                # we just accumulate it based on the logger. Note that the logger saves
                # per step, we will sum them up over the episode
                if step.get("J_damage", 0) > 0 or step.get("J_spatial", 0) > 0:
                    total_damage += step["J_damage"]
                    total_spatial += step["J_spatial"]
                    decision_steps += 1

        if decision_steps > 0:
            j_damage.append(total_damage)
            j_spatial.append(total_spatial)

    if not j_damage:
        print("No Pareto data found to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(j_damage, j_spatial, alpha=0.7, edgecolors='k', s=40)
    plt.xlabel('Degradation Cost (J_damage)')
    plt.ylabel('Spatial Logistics Cost (J_spatial)')
    plt.title('Pareto Front: Degradation vs Logistics')
    plt.tight_layout()

    if fmt in ["pdf", "both"]:
        plt.savefig(os.path.join(output_dir, "pareto_front.pdf"), format="pdf", dpi=300)
        print(f"Saved pareto front plot to {output_dir}/pareto_front.pdf")
    if fmt in ["png", "both"]:
        plt.savefig(os.path.join(output_dir, "pareto_front.png"), format="png", dpi=300)
        print(f"Saved pareto front plot to {output_dir}/pareto_front.png")
    plt.close()

def plot_health_index_trajectory(data, output_dir="plots", fmt="both"):
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("No data found for Health Index trajectory.")
        return

    plt.figure(figsize=(8, 4))

    # Plot for each episode
    for i, episode in enumerate(data):
        months = [step.get("elapsed_month", idx+1) for idx, step in enumerate(episode)]
        hi_mean = [step.get("Health Index", 0.0) for step in episode]

        # Plot individual trajectories with low alpha
        plt.plot(months, hi_mean, color='C0', alpha=0.3, linewidth=1.0)

    # Calculate and plot the average trajectory across all episodes
    if len(data) > 0:
        max_len = max(len(ep) for ep in data)
        avg_hi = np.zeros(max_len)
        counts = np.zeros(max_len)

        for episode in data:
            for j, step in enumerate(episode):
                avg_hi[j] += step.get("Health Index", 0.0)
                counts[j] += 1

        avg_hi = avg_hi / np.maximum(counts, 1)
        months_x = np.arange(1, max_len + 1)

        plt.plot(months_x, avg_hi, color='C1', linewidth=2.5, label='Average Health Index')

    plt.xlabel('Time (Months)')
    plt.ylabel('Mean Health Index (HI)')
    plt.title('Turbine Health Index Degradation Trajectory Over 25 Years')

    # Optional: adjust x-axis to years if preferred
    def months_to_years(x, pos):
        return f"{int(x/12)}"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(months_to_years))
    plt.xlabel('Time (Years)')

    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if fmt in ["pdf", "both"]:
        plt.savefig(os.path.join(output_dir, "hi_trajectory.pdf"), format="pdf", dpi=300)
        print(f"Saved health index trajectory plot to {output_dir}/hi_trajectory.pdf")
    if fmt in ["png", "both"]:
        plt.savefig(os.path.join(output_dir, "hi_trajectory.png"), format="png", dpi=300)
        print(f"Saved health index trajectory plot to {output_dir}/hi_trajectory.png")
    plt.close()

def plot_spatial_dispatch_map(env_config, sample_action, output_dir="plots", fmt="both"):
    import itertools
    os.makedirs(output_dir, exist_ok=True)

    try:
        x = env_config.spatial_grouping_objective.x
        y = env_config.spatial_grouping_objective.y
    except AttributeError:
        print("Cannot extract spatial grouping coordinates from env_config.")
        return

    plt.figure(figsize=(8, 8))

    selected_x = []
    selected_y = []

    for i in range(len(x)):
        if i < len(sample_action) and sample_action[i] == 1:
            plt.scatter(x[i], y[i], color='red', s=100, label='Serviced' if len(selected_x) == 0 else "")
            selected_x.append(x[i])
            selected_y.append(y[i])
        else:
            plt.scatter(x[i], y[i], color='blue', s=100, label='Unserviced' if i == 0 else "")

    # Draw dashed lines for mean pairwise spatial distance
    for (x1, y1), (x2, y2) in itertools.combinations(zip(selected_x, selected_y), 2):
        plt.plot([x1, x2], [y1, y2], 'k--', alpha=0.5)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Spatial Dispatch Map')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()

    if fmt in ["pdf", "both"]:
        plt.savefig(os.path.join(output_dir, "spatial_dispatch_map.pdf"), format="pdf", dpi=300)
        print(f"Saved spatial dispatch map to {output_dir}/spatial_dispatch_map.pdf")
    if fmt in ["png", "both"]:
        plt.savefig(os.path.join(output_dir, "spatial_dispatch_map.png"), format="png", dpi=300)
        print(f"Saved spatial dispatch map to {output_dir}/spatial_dispatch_map.png")
    plt.close()

def plot_action_heatmap(log_data, output_dir="plots", fmt="both"):
    os.makedirs(output_dir, exist_ok=True)

    wind_speeds = []
    health_indices = []

    for episode in log_data:
        for step in episode:
            if step.get("decision_event") == True:
                ambient_u = step.get("ambient_u")
                actions = step.get("action", [])
                hi_turbine = step.get("HI_turbine", [])

                if ambient_u is not None and actions and hi_turbine:
                    for a, hi in zip(actions, hi_turbine):
                        if a == 1:
                            wind_speeds.append(ambient_u)
                            health_indices.append(hi)

    if not wind_speeds:
        print("No action triggered data found for Action Heatmap.")
        return

    plt.figure(figsize=(8, 6))
    plt_sns.kdeplot(x=wind_speeds, y=health_indices, fill=True, cmap="YlOrRd", thresh=0, levels=20)
    plt.xlabel("Ambient Wind Speed [m/s]")
    plt.ylabel("Health Index (HI)")
    plt.title("Action Trigger Heatmap: Wind Speed vs Health Index")
    plt.tight_layout()

    if fmt in ["pdf", "both"]:
        plt.savefig(os.path.join(output_dir, "action_heatmap.pdf"), format="pdf", dpi=300)
        print(f"Saved action heatmap to {output_dir}/action_heatmap.pdf")
    if fmt in ["png", "both"]:
        plt.savefig(os.path.join(output_dir, "action_heatmap.png"), format="png", dpi=300)
        print(f"Saved action heatmap to {output_dir}/action_heatmap.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication quality plots from trajectories.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing JSON log files.")
    parser.add_argument("--out_dir", type=str, default="plots", help="Directory to save output PDFs.")
    parser.add_argument("--format", type=str, default="both", choices=["pdf", "png", "both"], help="Format to save the plots.")
    args = parser.parse_args()

    data = load_trajectories(args.log_dir)
    print(f"Loaded {len(data)} trajectories.")

    plot_pareto_front(data, args.out_dir, fmt=args.format)
    plot_health_index_trajectory(data, args.out_dir, fmt=args.format)
    plot_action_heatmap(data, args.out_dir, fmt=args.format)
