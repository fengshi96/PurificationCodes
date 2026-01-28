import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Path to data directory
data_dir = "clusterData/Kladder/chi200"

# Get all data files
data_files = sorted(glob.glob(os.path.join(data_dir, "finite_T_data_L*_h*.txt")))

# Extract h values from filenames and sort
h_values = []
data_by_h = {}

for fname in data_files:
    # Extract h value from filename: ...h0.05.txt (last occurrence of 'h')
    basename = os.path.basename(fname)
    # Find the last 'h' before '.txt'
    h_start = basename.rfind("_h") + 2  # +2 to skip "_h"
    h_end = basename.find(".txt")
    h_str = basename[h_start:h_end]
    h = float(h_str)
    h_values.append(h)
    
    # Read data: T | Sigmax | Sigmay | Sigmaz | S1 | S2 | Wp | Cv
    data = np.loadtxt(fname, skiprows=4)
    T = data[:, 0]
    Cv = data[:, 7]
    
    data_by_h[h] = {
        'T': T,
        'Cv': Cv
    }

# Sort h values
h_values = sorted(set(h_values))

# Exclude the last two h values (1.05 and 1.10)
h_values = h_values[:-2]

# Find the maximum T value (minimum common T) across all h values
# This ensures all h values are aligned to the same temperature range
T_min_common = max([data_by_h[h]['T'].min() for h in h_values])
T_max_common = min([data_by_h[h]['T'].max() for h in h_values])

print(f"Temperature range across all h: T_min = {T_min_common:.6f}, T_max = {T_max_common:.6f}")

# Trim all data to the common temperature range
for h in h_values:
    mask = (data_by_h[h]['T'] >= T_min_common) & (data_by_h[h]['T'] <= T_max_common)
    for key in data_by_h[h].keys():
        data_by_h[h][key] = data_by_h[h][key][mask]

# Create a grid of subplots
# For 20 h values, use 4 columns x 5 rows = 20 panels
ncols = 4
nrows = (len(h_values) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
axes = axes.flatten()  # Flatten to easily iterate

for idx, h in enumerate(h_values):
    ax1 = axes[idx]
    
    T = data_by_h[h]['T']
    Cv = data_by_h[h]['Cv']
    CvT = Cv / T
    
    # Plot Cv on primary y-axis
    color1 = 'tab:orange'
    ax1.set_xlabel(r'$T$', fontsize=10)
    ax1.set_ylabel(r'$C_v$', color=color1, fontsize=10)
    line1 = ax1.plot(T, Cv, 'o-', color=color1, linewidth=1.5, markersize=4, label=r'$C_v$')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=9)
    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=0.2)
    ax1.set_title(f'$h$ = {h:.2f}', fontsize=11, fontweight='bold')
    
    # Create secondary y-axis for Cv/T
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel(r'$C_v/T$', color=color2, fontsize=10)
    line2 = ax2.plot(T, CvT, 's-', color=color2, linewidth=1.5, markersize=4, label=r'$C_v/T$')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=9)
    # ax2.set_yscale('log')
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9)

# Hide unused subplots
for idx in range(len(h_values), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
save_path_cuts = os.path.join(data_dir, "Cv_cuts_grid.pdf")
plt.savefig(save_path_cuts, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_cuts}")
