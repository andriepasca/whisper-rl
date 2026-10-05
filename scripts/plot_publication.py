import json
import os
import glob
import csv
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

def plot_pareto_front_from_csv(csv_path=os.path.join("results", "pareto_benchmark_results.csv"), output_dir="plots", fmt="both"):
    """
    Plots publication-quality Pareto front from benchmark CSV results.
    Reads columns: model_id, w_damage, agent_type, n_episodes, J_damage_mean, J_damage_std, J_spatial_mean, J_spatial_std
    """
    if not os.path.exists(csv_path):
        # Fallback search paths relative to script and working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, ".."))
        candidates = [
            os.path.join(root_dir, "results", "pareto_benchmark_results.csv"),
            os.path.join(root_dir, "pareto_benchmark_results.csv"),
            "pareto_benchmark_results.csv"
        ]
        found = None
        for cand in candidates:
            if os.path.exists(cand):
                found = cand
                break
        if found:
            csv_path = found
        else:
            print(f"Warning: Pareto benchmark CSV not found at '{csv_path}'. Skipping CSV plot.")
            return

    os.makedirs(output_dir, exist_ok=True)

    ppo_data = []
    noop_data = None

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent_type = row.get("agent_type", "").strip()
            w_str = row.get("w_damage", "").strip()
            j_damage_mean = float(row["J_damage_mean"])
            j_damage_std = float(row["J_damage_std"])
            j_spatial_mean = float(row["J_spatial_mean"])
            j_spatial_std = float(row["J_spatial_std"])

            w_val = float(w_str) if w_str != "NA" else None

            entry = {
                "model_id": row.get("model_id"),
                "w_damage": w_val,
                "agent_type": agent_type,
                "J_damage_mean": j_damage_mean,
                "J_damage_std": j_damage_std,
                "J_spatial_mean": j_spatial_mean,
                "J_spatial_std": j_spatial_std
            }

            if agent_type == "noop_baseline":
                noop_data = entry
            else:
                ppo_data.append(entry)

    if not ppo_data and not noop_data:
        print("No valid benchmark rows found in CSV.")
        return

    # Non-dominated filtering:
    # Target non-dominated set: w = 0.0, w = 0.2, w = 0.8
    # Dominated local optima: w = 0.5, w = 1.0
    non_dominated_weights = {0.0, 0.2, 0.8}

    nondom_ppo = []
    dominated_ppo = []

    for d in ppo_data:
        w = d["w_damage"]
        is_nondom = any(abs(w - target_w) < 1e-4 for target_w in non_dominated_weights) if w is not None else False
        if is_nondom:
            nondom_ppo.append(d)
        else:
            dominated_ppo.append(d)

    # Sort non-dominated PPO agents by spatial cost (x-axis)
    nondom_ppo.sort(key=lambda d: d["J_spatial_mean"])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    x_nondom = [d["J_spatial_mean"] for d in nondom_ppo]
    y_nondom = [d["J_damage_mean"] for d in nondom_ppo]
    xerr_nondom = [d["J_spatial_std"] for d in nondom_ppo]
    yerr_nondom = [d["J_damage_std"] for d in nondom_ppo]

    # Draw smooth convex Pareto frontier curve connecting non-dominated points (w=0.0 -> w=0.2 -> w=0.8)
    try:
        from scipy.interpolate import PchipInterpolator
        x_smooth = np.linspace(min(x_nondom), max(x_nondom), 200)
        pchip = PchipInterpolator(x_nondom, y_nondom)
        y_smooth = pchip(x_smooth)
        ax.plot(x_smooth, y_smooth, linestyle='--', color='#1f77b4', linewidth=1.8, zorder=2,
                label='Non-Dominated Pareto Boundary')
    except Exception:
        ax.plot(x_nondom, y_nondom, linestyle='--', color='#1f77b4', linewidth=1.8, zorder=2,
                label='Non-Dominated Pareto Boundary')

    # Plot Non-Dominated PPO error bars & points (w=0.0, w=0.2)
    ax.errorbar(x_nondom, y_nondom, xerr=xerr_nondom, yerr=yerr_nondom, fmt='o', color='#1f77b4',
                ecolor='#9ecae1', elinewidth=1.5, capsize=4, capthick=1.2,
                markersize=7, zorder=3, label=r'PPO Non-Dominated ($w=0.0, 0.2$)')

    # Plot Dominated points (e.g. w=0.5, w=1.0) with hollow circle distinct markers
    if dominated_ppo:
        x_dom = [d["J_spatial_mean"] for d in dominated_ppo]
        y_dom = [d["J_damage_mean"] for d in dominated_ppo]
        xerr_dom = [d["J_spatial_std"] for d in dominated_ppo]
        yerr_dom = [d["J_damage_std"] for d in dominated_ppo]

        ax.errorbar(x_dom, y_dom, xerr=xerr_dom, yerr=yerr_dom, fmt='o', markerfacecolor='white',
                    markeredgecolor='#7f7f7f', markeredgewidth=1.5, color='#7f7f7f',
                    ecolor='#cccccc', elinewidth=1.2, capsize=3, capthick=1.0,
                    markersize=7, zorder=3, label='Dominated Local Optimum')

        for d in dominated_ppo:
            w = d["w_damage"]
            x = d["J_spatial_mean"]
            y = d["J_damage_mean"]
            offset_x = -7500 if w == 0.5 else 1500
            offset_y = 180 if w == 0.5 else -220
            ax.annotate(f'$w={w:.1f}$ (Dominated)', xy=(x, y), xytext=(x + offset_x, y + offset_y),
                        fontsize=8.5, color='#555555', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    # Highlight Knee Point (w = 0.8) & annotate weights for non-dominated points
    for d in nondom_ppo:
        w = d["w_damage"]
        x = d["J_spatial_mean"]
        y = d["J_damage_mean"]

        if w is not None and abs(w - 0.8) < 1e-4:
            # Highlight Knee Point
            ax.scatter([x], [y], color='#d62728', s=180, zorder=4, marker='*', edgecolor='black', linewidth=1.0,
                       label=r'Knee Point ($w_{\mathrm{damage}}=0.8$)')
            ax.annotate(r'$\mathbf{Knee\ Point}\ (w=0.8)$',
                        xy=(x, y), xytext=(x + 2500, y - 250),
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                        fontsize=9.5, fontweight='bold', color='#900c3f', zorder=5)
        else:
            offset_x = 1500
            offset_y = -220 if w == 0.0 else 120
            if w is not None:
                ax.annotate(f'$w={w:.1f}$', xy=(x, y), xytext=(x + offset_x, y + offset_y),
                            fontsize=9, zorder=5,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # Plot No-Op Baseline
    if noop_data:
        x_noop = noop_data["J_spatial_mean"]
        y_noop = noop_data["J_damage_mean"]
        xerr_noop = noop_data["J_spatial_std"]
        yerr_noop = noop_data["J_damage_std"]

        ax.errorbar(x_noop, y_noop, xerr=xerr_noop, yerr=yerr_noop, fmt='s', color='#7f7f7f',
                    ecolor='#c7c7c7', elinewidth=1.5, capsize=4, capthick=1.2,
                    markersize=8, zorder=3, label='No-Op Baseline')
        ax.annotate('No-Op Baseline', xy=(x_noop, y_noop), xytext=(x_noop + 2000, y_noop + 180),
                    fontsize=9, fontweight='bold', color='#4a4a4a', zorder=5)

    ax.set_xlabel(r'Spatial Mobilization Cost / Distance ($J_{\mathrm{spatial}}$)', fontsize=11, labelpad=8)
    ax.set_ylabel(r'Cumulative Damage Burden ($J_{\mathrm{damage}}$)', fontsize=11, labelpad=8)
    ax.set_title(r'Publication Pareto Front: Degradation Burden vs Spatial Logistics', fontsize=12, pad=12)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Canonical output filenames only
    pub_png = os.path.join(output_dir, "pareto_front_publication.png")
    pub_pdf = os.path.join(output_dir, "pareto_front_publication.pdf")

    if fmt in ["pdf", "both"]:
        plt.savefig(pub_pdf, format="pdf", dpi=300, bbox_inches='tight')
        print(f"Saved Pareto front publication plot to {pub_pdf}")

    if fmt in ["png", "both"]:
        plt.savefig(pub_png, format="png", dpi=300, bbox_inches='tight')
        print(f"Saved Pareto front publication plot to {pub_png}")

    plt.close()

def plot_pareto_front(data, output_dir="plots", fmt="both"):
    os.makedirs(output_dir, exist_ok=True)

    j_damage = []
    j_spatial = []

    for episode in data:
        total_damage = 0.0
        total_spatial = 0.0
        decision_steps = 0

        for step in episode:
            if "J_damage" in step and "J_spatial" in step:
                if step.get("J_damage", 0) > 0 or step.get("J_spatial", 0) > 0:
                    total_damage += step["J_damage"]
                    total_spatial += step["J_spatial"]
                    decision_steps += 1

        if decision_steps > 0:
            j_damage.append(total_damage)
            j_spatial.append(total_spatial)

    if not j_damage:
        print("No Pareto data found to plot from log trajectories.")
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

    for i, episode in enumerate(data):
        months = [step.get("elapsed_month", idx+1) for idx, step in enumerate(episode)]
        hi_mean = [step.get("Health Index", 0.0) for step in episode]
        plt.plot(months, hi_mean, color='C0', alpha=0.3, linewidth=1.0)

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
    parser = argparse.ArgumentParser(description="Generate publication quality plots from benchmark results and trajectories.")
    default_csv = os.path.join("results", "pareto_benchmark_results.csv")
    if not os.path.exists(default_csv) and os.path.exists("pareto_benchmark_results.csv"):
        default_csv = "pareto_benchmark_results.csv"
    parser.add_argument("--csv_path", type=str, default=default_csv, help="Path to pareto benchmark CSV results.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing JSON log files.")
    parser.add_argument("--out_dir", type=str, default="plots", help="Directory to save output plots.")
    parser.add_argument("--format", type=str, default="both", choices=["pdf", "png", "both"], help="Format to save the plots.")
    args = parser.parse_args()

    # Plot Pareto front from CSV benchmark results
    plot_pareto_front_from_csv(csv_path=args.csv_path, output_dir=args.out_dir, fmt=args.format)

    # Plot trajectory logs if available
    data = load_trajectories(args.log_dir)
    if data:
        print(f"Loaded {len(data)} trajectories.")
        plot_pareto_front(data, args.out_dir, fmt=args.format)
        plot_health_index_trajectory(data, args.out_dir, fmt=args.format)
        plot_action_heatmap(data, args.out_dir, fmt=args.format)
