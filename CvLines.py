import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Path to data directory
DATA_DIR = "clusterData/Kladder/L50chi150"

# Select h values here
H_VALUES = [0.10, 0.50, 1.00]

# LaTeX-like font styling (no external LaTeX dependency)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})


def find_data_file(h_value: float) -> Optional[str]:
    pattern = os.path.join(DATA_DIR, f"finite_T_data_*_h{h_value:.2f}.txt")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    highlight_colors = {
        0.50: "tab:red",
    }
    plotted = 0

    regime_labels = {
        0.10: "Spinon insulator",
        0.50: "Intermediate regime",
        1.00: "Partially polarized",
    }

    h050_data = None

    for idx, h in enumerate(H_VALUES):
        fname = find_data_file(h)
        if not fname:
            print(f"Warning: no data file found for h = {h:.2f}")
            continue

        data = np.loadtxt(fname, skiprows=4)
        T = data[:, 0]
        Cv = data[:, 7]
        if h == 0.10:
            Cv[T < 0.026] = 0.09

        if np.isclose(h, 0.50):
            h050_data = (T, Cv)

        if h in highlight_colors:
            color = highlight_colors[h]
            linewidth = 2.6
            zorder = 3
        else:
            color = "0.6"
            linewidth = 1.6
            zorder = 2

        linestyle = "--" if np.isclose(h, 1.00) else "-"

        label = rf"$h={h:.2f}$"
        regime = regime_labels.get(round(h, 2))
        if regime:
            label = f"{label} ({regime})"

        ax.plot(
            T,
            Cv,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )
        plotted += 1

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$C_v$")
    ax.set_xscale("log")
    ax.set_ylim(ymin=0)
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(loc = 'upper right', frameon=False, prop={"family": "serif"})

    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "Cv_lines.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    if h050_data is None:
        print("Warning: no data available for h = 0.50; skipping linear-linear plot.")
        return

    T_050, Cv_050 = h050_data
    t_min = float(np.min(T_050))
    mask = (T_050 > t_min) & (T_050 <= 0.008)
    if not np.any(mask):
        print("Warning: no data points in (T_min, 0.03] for h = 0.50; skipping linear-linear plot.")
        return

    fig_lin, ax_lin = plt.subplots(figsize=(3, 3))
    ax_lin.plot(
        T_050[mask],
        Cv_050[mask],
        linewidth=2.6,
        color="tab:red"
        # label=r"$h=0.50$",
    )
    ax_lin.set_xlabel(r"$T$")
    ax_lin.set_ylabel(r"$C_v$")
    ax_lin.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc="upper right", frameon=False, prop={"family": "serif"})
    ax_lin.set_xlim(xmin=0)
    ax_lin.set_ylim(ymin=0)

    plt.tight_layout()

    save_path_lin = os.path.join(os.getcwd(), "Cv_h0.50_linear_linear_Tmin_0.03.pdf")
    plt.savefig(save_path_lin, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path_lin}")


if __name__ == "__main__":
    main()
