import numpy as np
from model_Kladder import Kitaev_Ladder
import os
import h5py
# import pickle
import time
from tenpy.tools import hdf5_io
from tenpy.networks.mps import MPS, MPSEnvironment
from tenpy.networks.mpo import MPOEnvironment
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from mpo_current import CurrentOperators, current_op
# from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def perturb_state(psi, model_params, site=[0], chi_max=400):
    """ Perturb the state psi by applying a local energy density operator at site
    """
    op_params = model_params
    # del op_params['dmrg_restart']
    op_params['siteRef'] = site
    E_MPO = CurrentOperators(op_params).H_MPO 

    option = {
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }
    E_MPO.apply(psi, option)  # this alters psi in place
    print("[perturb_state] finished applying the energy density operator")



def measure_current_densities(env, lattice, model_params, site='all'):
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    ops_246_p, ops_357_m, ops_457_p, ops_345_p, ops_245_m, ops_456_m, \
        ops_53_p, ops_24_m, ops_75_p, ops_46_m, ops_24_p, ops_64_p, ops_75_m, ops_35_m = current_op(lattice, site)
    
    h = model_params['Fx'] # assuming hx = hy = hz = h
    exp_values = []

    print("-----------\n","Measuring the following current density operators:")
    for op_246_p, op_357_m, op_457_p, op_345_p, op_245_m, op_456_m, \
                        op_53_p, op_24_m, op_75_p, op_46_m, op_24_p, op_64_p, op_75_m, op_35_m \
            in zip(ops_246_p, ops_357_m, ops_457_p, ops_345_p, ops_245_m, ops_456_m, \
                        ops_53_p, ops_24_m, ops_75_p, ops_46_m, ops_24_p, ops_64_p, ops_75_m, ops_35_m):
        
        expvalue = (+2) * env.expectation_value_term(op_246_p) + (-2) * env.expectation_value_term(op_357_m) + env.expectation_value_term(op_457_p) \
                    + env.expectation_value_term(op_345_p) - env.expectation_value_term(op_245_m) - env.expectation_value_term(op_456_m) \
                    + (+h) * env.expectation_value_term(op_53_p) + (-h) *  env.expectation_value_term(op_24_m) + (+h) *  env.expectation_value_term(op_75_p) \
                    + (-h) * env.expectation_value_term(op_46_m) + (+h) *  env.expectation_value_term(op_24_p) + (+h) *  env.expectation_value_term(op_64_p) \
                    + (-h) *  env.expectation_value_term(op_75_m) + (-h) *  env.expectation_value_term(op_35_m)
        exp_values.append(expvalue) 
    

    return np.array(exp_values)





def measure_current_correlators(env, lattice, model_params, site):
    print("[measure_current_correlators] measuring energy correlators")
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list
    return measure_current_densities(env, lattice, model_params, site)



def measure_evolved_current(measurements, evolved_time, env, M, E0, model_params, site):
    """ function to measure energy density operators during time-evolution
    Parameters:
    measurements: thing to be measured
    evolved_time: evolved time
    evn: class MPSEnvironment(gs, psi)
    E0: ground state energy
    M: class Model
    
    Output Example:
    measurements = {
        t_key = 0.0: {
            'entropy': [...],
            'chi': [...],
            'Js': value,
            'Je': value
        },
        t_key = 0.1: {
            'entropy': [...],
            'chi': [...],
            'Js': value,
            'Je': value
        },
        ...
    }
    """
    t_key = np.round(evolved_time, 3)
    measurements[t_key] = {}

    if evolved_time == 0:
        measurements[t_key]['current_gs'] = measure_current_densities(env.bra, M.lat, model_params, site)
        current_gs_Ref = measurements[0]['current_gs'][model_params['Lx'] // 4 - 1]
        measurements[t_key]['elastic'] = np.array([current * current_gs_Ref for current in measurements[t_key]['current_gs']]) 

    print("-------------------- t_key or evolved time == ", t_key)

    measurements[t_key]['entropy'] = env.ket.entanglement_entropy()
    measurements[t_key]['chi'] = env.ket.chi
    measurements[t_key]['current_gs'] = measurements[0.00]['current_gs']

    measurements[t_key][f'correlator'] = np.exp(1j*evolved_time*E0) * measure_current_correlators(env,  M.lat, model_params, site)
    measurements[t_key][f'auto_correlator'] = measurements[t_key][f'correlator'] - measurements[0.00]['elastic']

    env.clear()


def run_time(**kwargs):
    """ run time evolution and measure hole correlation functions
    kwargs:
        gs_file: the name of the file that contains the ground state and model parameters
        op_type: operator type, e.g. Sigmaz, Sigmax, Sigmay
        j_unit_cell: location j of the operator, offset from the center
        dt: time step
        chi_max: bond dimension
        t_method: TDVP or ExpMPO
        tsteps_init: time span using SVD compression for ExpMPO
        tsteps_cont: time span using variational compression for ExpMPO
        save_state: whether we want to save the full states at some time steps (Default = False)
        evolve_gs: whether we want to also time evolve the ground state (Default = False)
    """
    # load ground state
    gs_file = kwargs['gs_file']  # file containing the ground state and model params
    name = os.path.basename(gs_file)

    with h5py.File(gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    run_time_restart = kwargs.get('run_time_restart', False)
    time_state_name = kwargs.get('time_state_name', None)
    # op_type = kwargs['op_type']
    model_params = state['model_params']
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    order = model_params.get('order', "default")
    print("order: ", order)

    if run_time_restart:
        time_state_name = kwargs['time_state_name']
        tsteps_init = 0
        tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP
        with h5py.File(time_state_name, 'r') as f:
            psi = hdf5_io.load_from_hdf5(f['last MPS'])
            phi = state['gs']
            measurements = hdf5_io.load_from_hdf5(f['measurements'])
            previous_evolved_time = hdf5_io.load_from_hdf5(f['evolved_time'])
            dt = hdf5_io.load_from_hdf5(f['dt'])
            previous_evolved_steps = int(previous_evolved_time/dt)
    else:
        tsteps_init = kwargs['tsteps_init'] # SVDs for ExpMPO, 2site for TDVP
        tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP
        previous_evolved_time = 0
        previous_evolved_steps = 0
        phi = state['gs']
        psi = phi.copy()
        measurements = {}



    # In case the gs is prepared polarized on the left edge,
    # we then focus on the quench dynamics with the Hamiltonian without the edge polarizing field
    M = Kitaev_Ladder(model_params)
    E0 = MPOEnvironment(phi, M.H_MPO, phi).full_contraction(0)


    chi_max = kwargs['chi_max']
    save_state = kwargs.get('save_state', False)  # whether we want to save the full states at some time steps
    evolve_gs = kwargs.get('evolve_gs', False)  # whether we want to also time evolve the ground state
    dt = kwargs['dt']
    t_method = kwargs['t_method']
    assert t_method == 'TDVP' or t_method == 'ExpMPO'
    
    if evolve_gs:
        t_name = f"_Energy_chi{chi_max}dt{dt:.3f}{t_method}_"+name
        E0 = 0  # don't need phase in correlations
    else:
        t_name = f"_Energy_chi{chi_max}dt{dt:.3f}{t_method}_"+name
    logger.info("name: " + t_name)

    # Parameters of ExpMPOEvolution(psi, model, t_params)
    # We set N_steps = 1, such that ExpMPOEvolution.run() evolves N_steps * dt = dt in time
    t_params = {'run_time_restart': run_time_restart, 'dt': dt, 'N_steps': 1, 'order': 2, 'approximation': 'II',
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
            'tsteps_init': tsteps_init, 'tsteps_cont': tsteps_cont,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }

    if t_method == 'ExpMPO':

        # if it is a brand new time-evolution from t=0
        purb_site = model_params['Lx'] - 5  # -5 for. E-odd ladder; -3 for E-even ladder
        if not run_time_restart:
            # then perturb by Ops at the site before time evolution by psi = \prod_j op_j |gs>
            print("Achtung!!!!!!!!!!!!!!!! I'm directly perturbing on the sites now:")
            print(purb_site)
            t_name = "_Pert_" + t_name
            if model_params['bc'] == 'open':
                # psi = perturb_state(psi, M.lat, model_params, purb_site)
                perturb_state(psi, model_params, site=purb_site, chi_max=chi_max)  # change psi=gs in place: psi = h_ref |gs>
            else:
                raise ValueError("PBC is not implemented yet!")

        else:
            print("Restart time evolution!")
            t_name = "_Restart_Pert_" + t_name

        eng = ExpMPOEvolution(psi, M, t_params)
        if run_time_restart:
            eng.evolved_time += previous_evolved_time



        # define sites to measure energy density operators
        if model_params['bc'] == 'open':
            site = range(2, M.lat.N_sites - 6, 4)
            print("site: ", site)
        else:
            raise ValueError("PBC is not implemented yet!")
        
        # stores the partial contractions between gs(bra) and psi(ket) up to each bond.
        # note that psi will be modified in-place
        env = MPSEnvironment(phi, psi)

        
        # measure the ground state expectation of energy
        if not run_time_restart:
            print("[main] measurementing expectation for t= 0 for the perturbed state")
            measure_evolved_current(measurements, eng.evolved_time, env, M, E0, model_params, site=site)
            print("[main] finished measurementing expectation for t= 0 for the perturbed state")

        # exp(-iHt) |psi> = exp(-iHt) h_ref |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_current(measurements, eng.evolved_time, env, M, E0, model_params, site=site)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)


        if not os.path.exists(f"./jcorrelator_states_{chi_max}/"):
            os.makedirs(f"./jcorrelator_states_{chi_max}/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_current(measurements, eng.evolved_time, env, M, E0, model_params, site=site)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
            print(measurements)

            # save state at the last step
            if i%100 == 0 or i == tsteps_cont - 1:  # i == tsteps_cont - 1: #i%2 == 0:
                state_t = {"gs_file": gs_file, "t_params": t_params, "fields": str(hx)+str(hy)+str(hz), "save_state": save_state, "order": order,
                           "chi_max": chi_max, "dt": dt, "t_method": t_method, "evolved_time": eng.evolved_time, "last MPS": psi.copy(), "model_params": model_params, "pos": M.positions(),
                           "measurements": measurements}
                if save_state:
                    with h5py.File(f'./jcorrelator_states_{chi_max}/{tsteps_init+previous_evolved_steps+i+1}'+f'time{round(eng.evolved_time, 2)}'+t_name, 'w') as f:
                        hdf5_io.save_to_hdf5(f, state_t)


    else:
        raise ValueError('So far only ExpMPO method is implemented')
    



# ----------------------------------------------------------
# ----------------- Debug run_time() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument

    run_time_restart = False
    
    total = len(sys.argv)

    # if total < 3:
    #     raise("missing arguments! at least 1 cmdargs gs_file!")
    # cmdargs = sys.argv

    gs_file = "./ground_states/GS_L13defaultchi150_K-1.00Fx-0.70Fy-0.70Fz-0.70.h5"
    # gs_file = cmdargs[1]
    # chi_max = int(cmdargs[2])

    previous_run_file = None
    if run_time_restart:
        previous_run_file = './energy_time_states/3time0.03_Pert_Sm_Energy_chi350dt0.010ExpMPO_GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5'

    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 1
    tsteps_cont = 1
    save_state = True
    evolve_gs = False
    chi_max = 150#300

    

    run_time(gs_file=gs_file, chi_max=chi_max, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,
             save_state=save_state, evolve_gs=evolve_gs, run_time_restart=run_time_restart, time_state_name=previous_run_file)