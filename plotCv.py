import numpy as np
import matplotlib.pyplot as plt

# Read data from file
data = np.loadtxt('Cv-J1J2.txt', skiprows=1)
T = data[:, 0]
Cv = data[:, 1]

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Cv on first axis
color1 = 'tab:orange'
ax1.set_xlabel(r'$T$', fontsize=12)
ax1.set_ylabel(r'$C_v$', color=color1, fontsize=12)
line1 = ax1.plot(T, Cv, 's-', color=color1, linewidth=2, label=r'$C_v$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
# ax1.set_yscale('log')
ax1.set_xlim(T.min(), 0.5)
ax1.grid(True, which='both', alpha=0.3)

# Create second y-axis for Cv/T
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel(r'$C_v/T$', color=color2, fontsize=12)
line2 = ax2.plot(T, Cv / T, '^-', color=color2, linewidth=2, label=r'$C_v/T$')
ax2.tick_params(axis='y', labelcolor=color2)
# ax2.set_yscale('log')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best', fontsize=11)

plt.tight_layout()
plt.savefig("Cv-J1J2.pdf", dpi=300, bbox_inches='tight')
print("Plot saved to Cv-J1J2.pdf")
plt.show()
