from model_Kladder import Kitaev_Ladder
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

    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {'trunc_params': trunc_params}
    beta = 0.
    if order == 1:
        Us = [M.H_MPO.make_U(-dt, approx)]
    elif order == 2:
        Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(psi, Us[0], options)

    S1s, S2s, Wps = [[x] for x in measure_fluxes(psi, lattice=M.lat, site=[0, 4, 8, 12, 16])]   
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
        s1, s2, wp = measure_fluxes(psi, lattice=M.lat, site=[0, 4, 8, 12, 16])
        S1s.append(s1)
        S2s.append(s2)
        Wps.append(wp)
        
        # Measure energy
        E = energy_beta(psi, M.H_MPO)
        Es.append(E)


    return {'beta': betas, 'Sz': Szs, 'S1': S1s, 'S2': S2s, 'Wp': Wps, 'E': Es}



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    data_mpo = imag_apply_mpo(h=0.1)

    import numpy as np
    import matplotlib.pyplot as plt
    print(data_mpo['S1'])
    
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
    axes[0, 0].set_ylim(-1, 1)
    axes[0, 0].legend(loc='best')
    
    # Plot 2: Fluxes
    axes[0, 1].plot(T_all, np.sum(data_mpo['S1'], axis=1) / 5, 'o-', label='S1')
    axes[0, 1].plot(T_all, np.sum(data_mpo['S2'], axis=1) / 5, 's-', label='S2')
    axes[0, 1].plot(T_all, np.sum(data_mpo['Wp'], axis=1) / 5, '^-', label='Wp')
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
