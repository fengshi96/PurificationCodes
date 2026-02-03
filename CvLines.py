import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Path to data directory
DATA_DIR = "clusterData/Kladder/chi200"

# Select h values here
H_VALUES = [0.0, 0.1, 0.50, 0.65, 1.00]

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
        0.65: "tab:blue",
    }
    plotted = 0

    regime_labels = {
        0.10: "spinon insulator",
        0.50: "gapless",
        0.65: "gapless",
        1.00: "partially polarized",
    }

    for idx, h in enumerate(H_VALUES):
        fname = find_data_file(h)
        if not fname:
            print(f"Warning: no data file found for h = {h:.2f}")
            continue

        data = np.loadtxt(fname, skiprows=4)
        T = data[:, 0]
        Cv = data[:, 7]

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
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(frameon=False, prop={"family": "serif"})

    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "Cv_lines.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    main()
