import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===== Configuration =====
chi_max = 270  # Bond dimension to use - CHANGE THIS TO SELECT DIFFERENT CHI
# ===== End Configuration =====

# Path to data directory
data_dir = "clusterData/J1J2"

# Get all data files for the specified chi value
data_files = sorted(glob.glob(os.path.join(data_dir, f"finite_T_data_*_chi{chi_max}_*conserve*.txt")))

if len(data_files) == 0:
    print(f"Error: No data files found for chi={chi_max} in {data_dir}")
    print(f"Searched for: {os.path.join(data_dir, f'finite_T_data_*_chi{chi_max}_*conserve*.txt')}")
    exit(1)

print(f"Found {len(data_files)} data files for chi={chi_max}")

# Extract beta values from filenames and organize data
beta_values = []
data_by_beta = {}

for fname in data_files:
    # Extract beta value from filename: ...beta100.0_...
    basename = os.path.basename(fname)
    beta_start = basename.find("_beta") + 5  # +5 to skip "_beta"
    beta_end = basename.find("_dt")
    beta_str = basename[beta_start:beta_end]
    beta = float(beta_str)
    beta_values.append(beta)
    
    # Read data: T | Sigmaz | Cv
    data = np.loadtxt(fname, skiprows=4)
    T = data[:, 0]
    
    # Check number of columns to handle both formats
    if data.shape[1] == 3:
        # J1J2 format: T | Sigmaz | Cv
        Sigmaz = data[:, 1]
        Cv = data[:, 2]
    else:
        # Fallback
        Sigmaz = data[:, 1]
        Cv = data[:, 2]
    
    data_by_beta[beta] = {
        'T': T,
        'Cv': Cv,
        'Sigmaz': Sigmaz
    }

# Sort beta values
beta_values = sorted(set(beta_values))

# ===== Figure 1: Cv and Cv/T vs T =====
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Define colors for Cv and Cv/T separately
colors_cv = plt.cm.Blues(np.linspace(0.4, 1, len(beta_values)))
colors_cvt = plt.cm.Reds(np.linspace(0.4, 1, len(beta_values)))

# Plot Cv on primary y-axis
color1 = 'tab:blue'
ax1.set_xlabel(r'$T$', fontsize=12)
ax1.set_ylabel(r'$C_v$', color=color1, fontsize=12)
ax1.set_xscale('log')
ax1.grid(True, which='both', alpha=0.2)

for idx, beta in enumerate(beta_values):
    T = data_by_beta[beta]['T']
    Cv = data_by_beta[beta]['Cv']
    ax1.plot(T, Cv, 'o-', color=colors_cv[idx], linewidth=1.5, markersize=4, 
             label=f'$C_v$ ($\\beta={beta:.1f}$)', alpha=0.8)

ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)

# Create secondary y-axis for Cv/T
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel(r'$C_v/T$', color=color2, fontsize=12)

for idx, beta in enumerate(beta_values):
    T = data_by_beta[beta]['T']
    Cv = data_by_beta[beta]['Cv']
    CvT = Cv / T
    ax2.plot(T, CvT, 's--', color=colors_cvt[idx], linewidth=1.5, markersize=3, 
             label=f'$C_v/T$ ($\\beta={beta:.1f}$)', alpha=0.8)

ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

fig1.suptitle(f'$C_v$ and $C_v/T$ vs T (chi={chi_max})', fontsize=13, fontweight='bold')
fig1.tight_layout()
save_path_cv = os.path.join(data_dir, f"Cv_vs_T_chi{chi_max}.pdf")
fig1.savefig(save_path_cv, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_cv}")

# ===== Figure 2: Sigmaz vs T =====
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Define colors for Sigmaz
colors_sigmaz = plt.cm.Greens(np.linspace(0.4, 1, len(beta_values)))

ax3.set_xlabel(r'$T$', fontsize=12)
ax3.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
ax3.set_xscale('log')
ax3.grid(True, which='both', alpha=0.2)

for idx, beta in enumerate(beta_values):
    T = data_by_beta[beta]['T']
    Sigmaz = data_by_beta[beta]['Sigmaz']
    ax3.plot(T, Sigmaz, 'o-', color=colors_sigmaz[idx], linewidth=1.5, markersize=4, 
             label=f'$\\beta={beta:.1f}$', alpha=0.8)

ax3.legend(loc='best', fontsize=10, title=f'chi={chi_max}')
ax3.set_ylim(-1, 1)
fig2.suptitle(f'$\\langle\\sigma_z\\rangle$ vs T (chi={chi_max})', fontsize=13, fontweight='bold')
fig2.tight_layout()
save_path_sigmaz = os.path.join(data_dir, f"Sigmaz_vs_T_chi{chi_max}.pdf")
fig2.savefig(save_path_sigmaz, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_sigmaz}")

