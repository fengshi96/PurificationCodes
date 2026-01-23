from model_J1J2 import J1J2
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

    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {'trunc_params': trunc_params}
    beta = 0.
    if order == 1:
        Us = [M.H_MPO.make_U(-dt, approx)]
    elif order == 2:
        Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(psi, Us[0], options)

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
        Szs.append(psi.expectation_value("Sigmaz"))
        
        # Measure energy
        E = energy_beta(psi, M.H_MPO)
        Es.append(E)


    return {'beta': betas, 'Sz': Szs, 'E': Es}



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    trunc_params = {'chi_max': 80, 'chi_min': 10, 'svd_min': 1.e-8}
    data_mpo = imag_apply_mpo(Lx=4, Ly=6, beta_max=20., J1=1.0, J2=0.125, Fz=0.0, conserve='Sz', trunc_params=trunc_params)

    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convert beta to temperature for all plots
    beta_all = np.array(data_mpo['beta'])
    T_all = 1.0 / beta_all
    T_all[0] = np.inf  # T at beta=0 is infinity
    
    N_sites = np.sum(data_mpo['Sz'][0]).size if isinstance(data_mpo['Sz'][0], np.ndarray) else len(data_mpo['Sz'][0])
    
    # Plot 1: Sigmaz
    axes[0, 0].plot(T_all, np.sum(data_mpo['Sz'], axis=1) / N_sites, 'o-', label='Sigmaz')
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
    plt.savefig("./finite-T-J1J2.pdf", dpi=300, bbox_inches='tight')
    
    # Save Cv vs T to ASCII file
    np.savetxt("Cv-J1J2.txt", np.column_stack([T, Cv]), header="T\tCv", fmt='%.6e')
    print("Saved Cv vs T to Cv-J1J2.txt")
