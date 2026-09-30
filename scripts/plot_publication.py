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

def plot_pareto_front(data, output_dir="plots"):
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
    plt.savefig(os.path.join(output_dir, "pareto_front.pdf"), format="pdf", dpi=300)
    plt.close()
    print(f"Saved pareto front plot to {output_dir}/pareto_front.pdf")

def plot_health_index_trajectory(data, output_dir="plots"):
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
    plt.savefig(os.path.join(output_dir, "hi_trajectory.pdf"), format="pdf", dpi=300)
    plt.close()
    print(f"Saved health index trajectory plot to {output_dir}/hi_trajectory.pdf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication quality plots from trajectories.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing JSON log files.")
    parser.add_argument("--out_dir", type=str, default="plots", help="Directory to save output PDFs.")
    args = parser.parse_args()

    data = load_trajectories(args.log_dir)
    print(f"Loaded {len(data)} trajectories.")

    plot_pareto_front(data, args.out_dir)
    plot_health_index_trajectory(data, args.out_dir)
