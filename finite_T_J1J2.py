from model_J1J2 import J1J2
import numpy as np
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
from tenpy.networks.mpo import MPOEnvironment
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def energy_beta(psi, H_mpo):
    """
    Compute <H> using MPOEnvironment.
    
    Inputs:
    psi: MPS or PurificationMPS state
    H_mpo: Hamiltonian MPO
    
    Output:
    E: <H> / <psi|psi>
    """
    envH = MPOEnvironment(psi, H_mpo, psi)
    num = envH.full_contraction(0)     # <psi|H|psi>
    den = 1 # psi.overlap(psi)             # <psi|psi>
    return (num / den).real


def imag_apply_mpo(Lx=6, Ly=6, beta_max=10., dt=0.05, order=2, bc="finite", approx="II", J1=1.0, J2=1.0, Fz=1e-5, conserve=None, trunc_params=None):
    # model_params for J1J2 model on triangular lattice
    model_params = dict(Lx=Lx, Ly=Ly, order='default', J1=J1, J2=J2, Fz=Fz, bc_MPS='finite', conserve=conserve)

    # Default truncation parameters if not provided
    if trunc_params is None:
        trunc_params = {'chi_max': 30, 'chi_min': 10, 'svd_min': 1.e-8}

    # run DMRG
    M = J1J2(model_params)
    chi_max = trunc_params['chi_max']
    
    # Total number of sites
    total_sites = Lx * Ly

    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {'trunc_params': trunc_params}
    beta = 0.
    if order == 1:
        Us = [M.H_MPO.make_U(-dt, approx)]
    elif order == 2:
        Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(psi, Us[0], options)

    # Note: With conserve='Sz', only Sigmaz is available
    # Sxs = [psi.expectation_value("Sigmax")]
    # Sys = [psi.expectation_value("Sigmay")]
    Szs = [psi.expectation_value("Sigmaz")]
    Es = [energy_beta(psi, M.H_MPO)]  # Initial energy at beta=0
    betas = [0.]

    while beta < beta_max:
        print("current beta = ", beta, " for beta_max = ", beta_max)
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        betas.append(beta)
        for U in Us:
            eng.init_env(U)  # reset environment, initialize new copy of psi
            eng.run()  # apply U to psi

        # Measure observables
        # Sxs.append(psi.expectation_value("Sigmax"))
        # Sys.append(psi.expectation_value("Sigmay"))
        Szs.append(psi.expectation_value("Sigmaz"))
        
        # Measure energy
        E = energy_beta(psi, M.H_MPO)
        Es.append(E)


    return {'beta': betas, 'Sz': Szs, 'E': Es, 
            'chi_max': chi_max, 'Lx': Lx, 'Ly': Ly, 'J1': J1, 'J2': J2, 'Fz': Fz, 'total_sites': total_sites}



def save_data_to_file(data_mpo, Lx, Ly, J1, J2, Fz, chi_max, filename):
    """
    Save measured observables vs temperature to ASCII text file.
    
    Inputs:
    data_mpo: dictionary containing beta, Sx, Sy, Sz, E, total_sites
    Lx, Ly: lattice dimensions
    J1, J2, Fz: model parameters
    chi_max: maximum bond dimension
    filename: output filename for saving data
    """
    # Extract data
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    # Get normalization factor from data
    total_sites = data_mpo['total_sites']
    
    # Average over sites (only Sigmaz is available with conserve='Sz')
    Sz_avg = np.sum(data_mpo['Sz'], axis=1) / total_sites
    E = np.array(data_mpo['E'])
    
    # Compute Cv from energy
    beta = beta_all[1:]
    E_trim = E[1:]
    dE = np.diff(E_trim)
    dbeta = np.diff(beta)
    Cv = -beta[:-1]**2 * dE / dbeta
    T_Cv = 1.0 / beta[:-1]
    
    # Prepare header
    header = (f"Finite Temperature Data for J1J2 Model\n"
              f"Lx={Lx}, Ly={Ly}, J1={J1:.4f}, J2={J2:.4f}, Fz={Fz:.4f}, chi_max={chi_max}\n"
              f"Columns: T | Sigmaz | Cv\n")
    
    # Prepare data for saving (align by using only common T values)
    # Cv has one fewer point, so we save Cv data only
    data_to_save = np.column_stack((
        T_Cv,
        Sz_avg[2:],  # skip first two points to match Cv length
        Cv
    ))
    
    np.savetxt(filename, data_to_save, 
               header=header,
               fmt='%.6e',
               delimiter='\t',
               comments='# ')
    
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    """
    Main script for finite temperature TEBD simulation of J1J2 model.
    
    This script performs imaginary time evolution using PurificationTEBD to study finite temperature properties
    of the J1J2 model on a triangular lattice. It measures sigma operators (Sigmax, Sigmay, Sigmaz)
    as a function of temperature (inverse temperature beta).
    
    Command-line arguments:
    - --Lx: Lattice dimension in x direction (default: 6)
    - --Ly: Lattice dimension in y direction (default: 6)
    - --beta_max: Maximum inverse temperature for the simulation (default: 10.0)
    - --dt: Time step for the TEBD evolution (default: 0.025)
    - --J1: Nearest-neighbor coupling (default: 1.0)
    - --J2: Next-nearest-neighbor coupling (default: 0.125)
    - --Fz: Magnetic field strength (default: 0.0)
    - --chi_max: Maximum bond dimension for MPS (default: 30)
    
    Output:
    - Saves measured data (T, Sigmax, Sigmay, Sigmaz, Cv) to ASCII text file
    - Generates and saves a matplotlib figure with 4 subplots showing temperature dependence
    
    Example usage:
    $ python finite_T_J1J2.py --Lx 6 --Ly 6 --beta_max 20.0 --J1 1.0 --J2 0.125 --chi_max 100
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Finite temperature purification simulation for J1J2 model')
    parser.add_argument('--Lx', type=int, default=6, help='Lattice dimension in x direction (default: 6)')
    parser.add_argument('--Ly', type=int, default=6, help='Lattice dimension in y direction (default: 6)')
    parser.add_argument('--beta_max', type=float, default=10., help='Maximum inverse temperature (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.025, help='Time step (default: 0.025)')
    parser.add_argument('--J1', type=float, default=1.0, help='Nearest-neighbor coupling (default: 1.0)')
    parser.add_argument('--J2', type=float, default=0.125, help='Next-nearest-neighbor coupling (default: 0.125)')
    parser.add_argument('--Fz', type=float, default=0.0, help='Magnetic field strength (default: 0.0)')
    parser.add_argument('--chi_max', type=int, default=30, help='Maximum bond dimension (default: 30)')
    
    args = parser.parse_args()
    
    trunc_params = {'chi_max': args.chi_max, 'chi_min': 10, 'svd_min': 1.e-8}
    data_mpo = imag_apply_mpo(Lx=args.Lx, Ly=args.Ly, dt=args.dt, beta_max=args.beta_max, J1=args.J1, J2=args.J2, Fz=args.Fz, conserve='Sz', trunc_params=trunc_params)

    import matplotlib.pyplot as plt
    
    # Save data to file
    filename = f"Data_J1J2/finite_T_data_Lx{args.Lx}_Ly{args.Ly}_beta{args.beta_max:.1f}_dt{args.dt:.4f}_chi{data_mpo['chi_max']}_J1{args.J1:.2f}_J2{args.J2:.4f}_Fz{args.Fz:.2f}.txt"
    save_data_to_file(data_mpo, Lx=args.Lx, Ly=args.Ly, J1=args.J1, J2=args.J2, Fz=args.Fz, chi_max=data_mpo['chi_max'], filename=filename)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convert beta to temperature for all plots
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    total_sites = data_mpo['total_sites']
    
    # Plot 1: Sigmaz
    axes[0, 0].plot(T_all, np.sum(data_mpo['Sz'], axis=1) / total_sites, 'o-', label='Sigmaz')
    axes[0, 0].set_xlabel(r'$T$')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_ylabel(r'$\langle \sigma^z \rangle$')
    axes[0, 0].set_ylim(-1, 1)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid()
    
    # Plot 2: Internal Energy
    axes[0, 1].plot(T_all, data_mpo['E'], 'o-', label='E')
    axes[0, 1].set_xlabel(r'$T$')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylabel(r'$\langle H \rangle$')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid()
    
    # Plot 3: Specific Heat (computed from energy)
    beta = beta_all[1:]  # skip beta=0
    E = np.array(data_mpo['E'])[1:]  # skip first point
    # Cv = -beta^2 * (dE/dbeta)
    # Simple finite difference for dE/dbeta
    dE = np.diff(E)
    dbeta = np.diff(beta)
    Cv = - beta[:-1]**2 * dE / dbeta  # specific heat
    T = 1.0 / beta[:-1]  # Convert beta to temperature (in units where k_B = 1)
    
    ax1 = axes[1, 0]
    ax1.plot(T, Cv, 's-', label='Cv', color='orange', linewidth=2)
    ax1.set_xlabel(r'$T$')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$C_v$', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.legend(loc='best')
    ax1.grid()
    
    # Plot 4: Cv/T
    ax2 = axes[1, 1]
    ax2.plot(T, Cv / T, '^-', label='Cv/T', color='green', linewidth=2)
    ax2.set_xlabel(r'$T$')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$C_v/T$', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='best')
    ax2.grid()

    plt.tight_layout()
    plt.savefig("Data_J1J2/finite-T-J1J2.pdf", dpi=300, bbox_inches='tight')
