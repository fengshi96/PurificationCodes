"""
Scan finite-T calculations over different magnetic field strengths.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import importlib.util
import sys

# Import finite_T module (which has a hyphen in filename)
spec = importlib.util.spec_from_file_location("finite_T", "finite-T.py")
finite_T = importlib.util.module_from_spec(spec)
sys.modules["finite_T"] = finite_T
spec.loader.exec_module(finite_T)

imag_apply_mpo = finite_T.imag_apply_mpo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_data(data_mpo, h, filename):
    """
    Generate the same 4-panel plot as in finite_T.py
    
    Inputs:
    data_mpo: dict with 'beta', 'Sz', 'S1', 'S2', 'Wp', 'E' keys
    h: magnetic field value
    filename: output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convert beta to temperature for all plots
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    # Plot 1: Sigmaz
    axes[0, 0].plot(T_all, np.sum(data_mpo['Sz'], axis=1) / 22, 'o-', label='Sigmaz')
    axes[0, 0].set_xlabel(r'$T$')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_ylabel(r'$\langle \sigma^z \rangle$')
    axes[0, 0].set_ylim(-1, 0)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid()
    
    # Plot 2: Fluxes
    axes[0, 1].plot(T_all, np.sum(data_mpo['S1'], axis=1) / 5, 'o-', label='S1')
    axes[0, 1].plot(T_all, np.sum(data_mpo['S2'], axis=1) / 5, 's-', label='S2')
    axes[0, 1].plot(T_all, np.sum(data_mpo['Wp'], axis=1) / 5, '^-', label='Wp')
    axes[0, 1].set_xlabel(r'$T$')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylabel('Flux')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid()
    
    # Plot 3: Internal Energy
    axes[1, 0].plot(T_all, data_mpo['E'], 'o-', label='E')
    axes[1, 0].set_xlabel(r'$T$')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel(r'$\langle H \rangle$')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid()
    
    # Plot 4: Specific Heat (computed from energy)
    beta = beta_all[1:]  # skip beta=0
    E = np.array(data_mpo['E'])[1:]  # skip first point
    # Cv = -beta^2 * (dE/dbeta)
    # Simple finite difference for dE/dbeta
    dE = np.diff(E)
    dbeta = np.diff(beta)
    Cv = - beta[:-1]**2 * dE / dbeta  # specific heat
    T = 1.0 / beta[:-1]  # Convert beta to temperature (in units where k_B = 1)
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(T, Cv, 's-', label='Cv', color='orange', linewidth=2)
    line2 = ax2.plot(T, Cv / T, '^-', label='Cv/T', color='green', linewidth=2)
    
    ax1.set_xlabel(r'$T$')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$C_v$', color='orange')
    ax2.set_ylabel(r'$C_v/T$', color='green')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid()
    
    fig.suptitle(f'h = {h:.2f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")
    
    return T, Cv


if __name__ == "__main__":
    # Scan over magnetic field values
    h_values = np.array([1e-5, 0.10, 0.20, 0.53, 0.57, 0.62, 0.67, 0.72, 0.76, 0.81])
    
    # Store results
    results = {}
    
    for h in h_values:
        logger.info(f"Running simulation for h = {h:.2f}")
        data_mpo = imag_apply_mpo(beta_max=120, h=h)
        
        # Generate filename
        filename = f"finite-T-fine-h{h:.2f}.pdf"
        
        # Plot and save, get T and Cv
        T, Cv = plot_data(data_mpo, h, filename)
        results[h] = (T, Cv)
    
    # Save all Cv data to files
    for h, (T, Cv) in results.items():
        txt_filename = f"Cv-h{h:.2f}.txt"
        np.savetxt(txt_filename, np.column_stack([T, Cv]), fmt='%.6e', header='T\tCv')
        logger.info(f"Saved: {txt_filename}")
    
    logger.info("Scan complete!")
