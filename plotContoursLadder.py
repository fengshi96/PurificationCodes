import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os

def safe_log10(arr):
    out = np.full_like(arr, np.nan, dtype=float)
    mask = arr > 0
    out[mask] = np.log10(arr[mask])
    return out

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
    Sigmax = data[:, 1]
    Sigmay = data[:, 2]
    Sigmaz = data[:, 3]
    S1 = data[:, 4]
    S2 = data[:, 5]
    Wp = data[:, 6]
    Cv = data[:, 7]
    
    data_by_h[h] = {
        'T': T,
        'Sigmax': Sigmax,
        'Sigmay': Sigmay,
        'Sigmaz': Sigmaz,
        'S1': S1,
        'S2': S2,
        'Wp': Wp,
        'Cv': Cv
    }

# Sort h values
h_values = sorted(set(h_values))

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

# Create mesh grid for contour plots
# Build 2D arrays: h x T
h_mesh = []
T_mesh = []
Cv_mesh = []
CvT_mesh = []
Sigmax_mesh = []
Sigmay_mesh = []
Sigmaz_mesh = []
S1_mesh = []
S2_mesh = []
Wp_mesh = []

T_ref = data_by_h[h_values[0]]['T']  # Reference temperature array

for h in h_values:
    h_mesh.append(h)
    T_mesh.append(T_ref)
    Cv_mesh.append(data_by_h[h]['Cv'])
    CvT_mesh.append(data_by_h[h]['Cv'] / T_ref)
    Sigmax_mesh.append(data_by_h[h]['Sigmax'])
    Sigmay_mesh.append(data_by_h[h]['Sigmay'])
    Sigmaz_mesh.append(data_by_h[h]['Sigmaz'])
    S1_mesh.append(data_by_h[h]['S1'])
    S2_mesh.append(data_by_h[h]['S2'])
    Wp_mesh.append(data_by_h[h]['Wp'])

# Convert to 2D arrays
H, T = np.meshgrid(h_values, T_ref)
Cv_2d = np.array(Cv_mesh).T
CvT_2d = np.array(CvT_mesh).T
Sigmax_2d = np.array(Sigmax_mesh).T
Sigmay_2d = np.array(Sigmay_mesh).T
Sigmaz_2d = np.array(Sigmaz_mesh).T
S1_2d = np.array(S1_mesh).T
S2_2d = np.array(S2_mesh).T
Wp_2d = np.array(Wp_mesh).T

# ===== Figure 1: Cv and Cv/T contour plots =====
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# Cv contour
log_Cv_2d = safe_log10(Cv_2d)
finite_cv = np.isfinite(log_Cv_2d)
levels_cv = np.linspace(log_Cv_2d[finite_cv].min()/20, log_Cv_2d[finite_cv].max()*1.1, 100)
cf1 = axes1[0].contourf(H, T, log_Cv_2d, levels=levels_cv, cmap='jet', extend='both')
axes1[0].set_xlabel(r'$h$', fontsize=12)
axes1[0].set_ylabel(r'$T$', fontsize=12)
axes1[0].set_title(r'$C_v$ vs $(h, T)$', fontsize=12)
axes1[0].set_yscale('log')
cbar1 = plt.colorbar(cf1, ax=axes1[0], extend='both')
cbar1.set_label(r'$C_v$', fontsize=11)

# Cv/T contour
log_CvT_2d = safe_log10(CvT_2d)
finite_cvt = np.isfinite(log_CvT_2d)
levels_cvt = np.linspace(log_CvT_2d[finite_cvt].min()/20, log_CvT_2d[finite_cvt].max()*1.1, 100)
cf2 = axes1[1].contourf(H, T, log_CvT_2d, levels=levels_cvt, cmap='jet', extend='both')
axes1[1].set_xlabel(r'$h$', fontsize=12)
axes1[1].set_ylabel(r'$T$', fontsize=12)
axes1[1].set_title(r'$C_v/T$ vs $(h, T)$', fontsize=12)
axes1[1].set_yscale('log')
cbar2 = plt.colorbar(cf2, ax=axes1[1], extend='both')
cbar2.set_label(r'$C_v/T$', fontsize=11)

plt.tight_layout()
save_path_cv = os.path.join(data_dir, "contour_Cv.pdf")
plt.savefig(save_path_cv, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_cv}")

# ===== Figure 2: S1, S2, Wp contour plots =====
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# S1 contour
levels_s1 = np.linspace(S1_2d.min(), S1_2d.max(), 50)
cf3 = axes2[0].contourf(H, T, S1_2d, levels=levels_s1, cmap='RdYlBu_r', extend='both')
axes2[0].set_xlabel(r'$h$', fontsize=12)
axes2[0].set_ylabel(r'$T$', fontsize=12)
axes2[0].set_title(r'$S_1$ vs $(h, T)$', fontsize=12)
axes2[0].set_yscale('log')
cbar3 = plt.colorbar(cf3, ax=axes2[0], extend='both')
cbar3.set_label(r'$S_1$', fontsize=11)

# S2 contour
levels_s2 = np.linspace(S2_2d.min(), S2_2d.max(), 50)
cf4 = axes2[1].contourf(H, T, S2_2d, levels=levels_s2, cmap='RdYlBu_r', extend='both')
axes2[1].set_xlabel(r'$h$', fontsize=12)
axes2[1].set_ylabel(r'$T$', fontsize=12)
axes2[1].set_title(r'$S_2$ vs $(h, T)$', fontsize=12)
axes2[1].set_yscale('log')
cbar4 = plt.colorbar(cf4, ax=axes2[1], extend='both')
cbar4.set_label(r'$S_2$', fontsize=11)

# Wp contour
levels_wp = np.linspace(Wp_2d.min(), Wp_2d.max(), 50)
cf5 = axes2[2].contourf(H, T, Wp_2d, levels=levels_wp, cmap='RdYlBu_r', extend='both')
axes2[2].set_xlabel(r'$h$', fontsize=12)
axes2[2].set_ylabel(r'$T$', fontsize=12)
axes2[2].set_title(r'$W_p$ vs $(h, T)$', fontsize=12)
axes2[2].set_yscale('log')
cbar5 = plt.colorbar(cf5, ax=axes2[2], extend='both')
cbar5.set_label(r'$W_p$', fontsize=11)

plt.tight_layout()
save_path_fluxes = os.path.join(data_dir, "contour_fluxes.pdf")
plt.savefig(save_path_fluxes, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_fluxes}")

# ===== Figure 3: Sigmax, Sigmay, Sigmaz contour plots =====
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

# Sigmax contour
levels_sx = np.linspace(-.5, 0, 50)
cf6 = axes3[0].contourf(H, T, Sigmax_2d, levels=levels_sx, cmap='coolwarm_r', extend='both')
axes3[0].set_xlabel(r'$h$', fontsize=12)
axes3[0].set_ylabel(r'$T$', fontsize=12)
axes3[0].set_title(r'$\langle\sigma_x\rangle$ vs $(h, T)$', fontsize=12)
axes3[0].set_yscale('log')
cbar6 = plt.colorbar(cf6, ax=axes3[0], extend='both')
cbar6.set_label(r'$\langle\sigma_x\rangle$', fontsize=11)

# Sigmay contour
levels_sy = np.linspace(-.5, 0, 50)
cf7 = axes3[1].contourf(H, T, Sigmay_2d, levels=levels_sy, cmap='coolwarm_r', extend='both')
axes3[1].set_xlabel(r'$h$', fontsize=12)
axes3[1].set_ylabel(r'$T$', fontsize=12)
axes3[1].set_title(r'$\langle\sigma_y\rangle$ vs $(h, T)$', fontsize=12)
axes3[1].set_yscale('log')
cbar7 = plt.colorbar(cf7, ax=axes3[1], extend='both')
cbar7.set_label(r'$\langle\sigma_y\rangle$', fontsize=11)

# Sigmaz contour
levels_sz = np.linspace(-.5, 0, 50)
cf8 = axes3[2].contourf(H, T, Sigmaz_2d, levels=levels_sz, cmap='coolwarm_r', extend='both')
axes3[2].set_xlabel(r'$h$', fontsize=12)
axes3[2].set_ylabel(r'$T$', fontsize=12)
axes3[2].set_title(r'$\langle\sigma_z\rangle$ vs $(h, T)$', fontsize=12)
axes3[2].set_yscale('log')
cbar8 = plt.colorbar(cf8, ax=axes3[2], extend='both')
cbar8.set_label(r'$\langle\sigma_z\rangle$', fontsize=11)

plt.tight_layout()
save_path_mag = os.path.join(data_dir, "contour_magnetization.pdf")
plt.savefig(save_path_mag, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path_mag}")
