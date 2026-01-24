from models.model_Kladder import Kitaev_Ladder
import numpy as np
from mpo_fluxes_Kladder import S1Operators, S2Operators, fluxes_op
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




def measure_fluxes(psi, lattice, site='all'):
    """
    Measure flux operators for a given psi.
    For a ladder, there are three types of fluxes: S1 (square faces a), S2 (square faces b) and Wp (hexagons).
    
    Inputs:
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available energy density operators are matched in terms of operators' site of reference

    Output:
    expectation value <psi| flux_op |psi>
    """
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    ops_s1, ops_s2, ops_wp = fluxes_op(lattice, site=site)
    exp_values_s1 = []
    exp_values_s2 = []
    exp_values_wp = []
    
    for op_s1 in ops_s1:
        # print(op_s1)
        expvalue = psi.expectation_value_term(op_s1) + 0.
        exp_values_s1.append(expvalue) 

    for op_s2 in ops_s2:
        # print(op_s2)
        expvalue = psi.expectation_value_term(op_s2) + 0.
        exp_values_s2.append(expvalue) 

    for op_wp in ops_wp:
        # print(op_wp)
        expvalue = psi.expectation_value_term(op_wp) + 0.
        exp_values_wp.append(expvalue) 
    
    return exp_values_s1, exp_values_s2, exp_values_wp



def imag_apply_mpo(L=11, beta_max=10., dt=0.05, order=2, bc="finite", approx="II", h=1e-5, trunc_params=None):
    # model_params = dict(Lx=L, order='default', J_K=1, Fx=1e-5, Fy=1e-5, Fz=1e-5, bc='open')
    model_params = dict(Lx=L, order='default', J_K=-1, Fx=h, Fy=h, Fz=h, bc='open')

    # Default truncation parameters if not provided
    if trunc_params is None:
        trunc_params = {'chi_max': 30, 'chi_min': 10, 'svd_min': 1.e-8}

    # run DMRG
    M = Kitaev_Ladder(model_params)
    chi_max = trunc_params['chi_max']

    # Automatically determine flux measurement sites and normalization
    # For a ladder with L legs, there are 2*L total sites
    total_sites = 2 * L
    # Number of Wp operators (hexagon plaquettes) under open boundary conditions
    num_flux_ops = (L - 1) // 2
    # Sites where to measure fluxes (left-bottom site of each Wp hexagon)
    flux_sites = [4 * i for i in range(num_flux_ops)]
    print(flux_sites)

    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {'trunc_params': trunc_params}
    beta = 0.
    if order == 1:
        Us = [M.H_MPO.make_U(-dt, approx)]
    elif order == 2:
        Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(psi, Us[0], options)

    S1s, S2s, Wps = [[x] for x in measure_fluxes(psi, lattice=M.lat, site=flux_sites)]   
    Sxs = [psi.expectation_value("Sigmax")]
    Sys = [psi.expectation_value("Sigmay")]
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
        Sxs.append(psi.expectation_value("Sigmax"))
        Sys.append(psi.expectation_value("Sigmay"))
        Szs.append(psi.expectation_value("Sigmaz"))
        s1, s2, wp = measure_fluxes(psi, lattice=M.lat, site=flux_sites)
        S1s.append(s1)
        S2s.append(s2)
        Wps.append(wp)
        
        # Measure energy
        E = energy_beta(psi, M.H_MPO)
        Es.append(E)


    return {'beta': betas, 'Sx': Sxs, 'Sy': Sys, 'Sz': Szs, 'S1': S1s, 'S2': S2s, 'Wp': Wps, 'E': Es, 
            'chi_max': chi_max, 'h': h, 'L': L, 'total_sites': total_sites, 'num_flux_ops': num_flux_ops}



def save_data_to_file(data_mpo, h, chi_max, filename):
    """
    Save measured observables vs temperature to ASCII text file.
    
    Inputs:
    data_mpo: dictionary containing beta, Sx, Sy, Sz, S1, S2, Wp, E, total_sites, num_flux_ops
    h: magnetic field value
    chi_max: maximum bond dimension
    filename: output filename for saving data
    """
    import numpy as np
    
    # Extract data
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    # Get normalization factors from data
    total_sites = data_mpo['total_sites']
    num_flux_ops = data_mpo['num_flux_ops']
    
    # Average over sites
    Sx_avg = np.sum(data_mpo['Sx'], axis=1) / total_sites
    Sy_avg = np.sum(data_mpo['Sy'], axis=1) / total_sites
    Sz_avg = np.sum(data_mpo['Sz'], axis=1) / total_sites
    S1_avg = np.sum(data_mpo['S1'], axis=1) / num_flux_ops
    S2_avg = np.sum(data_mpo['S2'], axis=1) / num_flux_ops
    Wp_avg = np.sum(data_mpo['Wp'], axis=1) / num_flux_ops
    E = np.array(data_mpo['E'])
    
    # Compute Cv from energy
    beta = beta_all[1:]
    E_trim = E[1:]
    dE = np.diff(E_trim)
    dbeta = np.diff(beta)
    Cv = -beta[:-1]**2 * dE / dbeta
    T_Cv = 1.0 / beta[:-1]
    
    # Prepare header
    header = (f"Finite Temperature Data for Kitaev Ladder\n"
              f"chi_max = {chi_max}, h = {h:.4f}\n"
              f"Columns: T | Sigmax | Sigmay | Sigmaz | S1 | S2 | Wp | Cv\n")
    
    # Prepare data for saving (align by using only common T values)
    # Cv has one fewer point, so we save Cv data only
    data_to_save = np.column_stack((
        T_Cv,
        Sx_avg[2:],  # skip first two points to match Cv length
        Sy_avg[2:],
        Sz_avg[2:],
        S1_avg[2:],
        S2_avg[2:],
        Wp_avg[2:],
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
    Main script for finite temperature TEBD simulation of Kitaev Ladder.
    
    This script performs imaginary time evolution using PurificationTEBD to study finite temperature properties
    of the Kitaev Ladder model. It measures various observables (Sigmax, Sigmay, Sigmaz, flux operators S1, S2, Wp)
    as a function of temperature (inverse temperature beta).
    
    Command-line arguments:
    - --L: Number of legs in the ladder (default: 11)
    - --beta_max: Maximum inverse temperature for the simulation (default: 10.0)
    - --dt: Time step for the TEBD evolution (default: 0.025)
    - --h: Magnetic field strength (default: 0.1)
    - --chi_max: Maximum bond dimension for MPS (default: 30)
    
    Output:
    - Saves measured data (T, Sigmax, Sigmay, Sigmaz, S1, S2, Wp, Cv) to ASCII text file
      with filename: some_dir/finite_T_data_chi{chi_max}_h{h:.2f}.txt
    
    Example usage:
    $ python finite_T_Kladder.py --L 11 --beta_max 10.0 --h 0.1 --chi_max 30
    """
    import argparse
    
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Finite temperature purification simulation for Kitaev Ladder')
    parser.add_argument('--L', type=int, default=11, help='Number of legs in the ladder (default: 11)')
    parser.add_argument('--beta_max', type=float, default=10., help='Maximum inverse temperature (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.025, help='Time step (default: 0.025)')
    parser.add_argument('--h', type=float, default=0.1, help='Magnetic field strength (default: 0.1)')
    parser.add_argument('--chi_max', type=int, default=30, help='Maximum bond dimension (default: 30)')
    
    args = parser.parse_args()

    trunc_params = {'chi_max': args.chi_max, 'chi_min': 10, 'svd_min': 1.e-8}
    data_mpo = imag_apply_mpo(L=args.L, beta_max=args.beta_max, dt=args.dt, order=2, bc="finite", approx="II", h=args.h, 
                              trunc_params=trunc_params)

    
    # Save data to file
    filename = f"Data_AFM_Kitaev_Ladder/finite_T_data_L{args.L}_beta{args.beta_max:.1f}_dt{args.dt:.4f}_chi{data_mpo['chi_max']}_h{data_mpo['h']:.2f}.txt"
    save_data_to_file(data_mpo, h=data_mpo['h'], chi_max=data_mpo['chi_max'], filename=filename)
    
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convert beta to temperature for all plots
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    # Get normalization factors
    total_sites = data_mpo['total_sites']
    num_flux_ops = data_mpo['num_flux_ops']
    
    # Plot 1: Sigmaz
    axes[0, 0].plot(T_all, np.sum(data_mpo['Sz'], axis=1) / total_sites, 'o-', label='Sigmaz')
    axes[0, 0].set_xlabel(r'$T$')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_ylabel(r'$\langle \sigma^z \rangle$')
    axes[0, 0].set_ylim(-1, 1)
    axes[0, 0].legend(loc='best')
    
    # Plot 2: Fluxes
    axes[0, 1].plot(T_all, np.sum(data_mpo['S1'], axis=1) / num_flux_ops, 'o-', label='S1')
    axes[0, 1].plot(T_all, np.sum(data_mpo['S2'], axis=1) / num_flux_ops, 's-', label='S2')
    axes[0, 1].plot(T_all, np.sum(data_mpo['Wp'], axis=1) / num_flux_ops, '^-', label='Wp')
    axes[0, 1].set_xlabel(r'$T$')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylabel('Flux')
    axes[0, 1].legend(loc='best')
    
    # Plot 3: Internal Energy
    axes[1, 0].plot(T_all, data_mpo['E'], 'o-', label='E')
    axes[1, 0].set_xlabel(r'$T$')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel(r'$\langle H \rangle$')
    axes[1, 0].legend(loc='best')
    
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

    plt.tight_layout()
    plt.savefig("./Data_AFM_Kitaev_Ladder/finite-T-fine.pdf", dpi=300, bbox_inches='tight')
