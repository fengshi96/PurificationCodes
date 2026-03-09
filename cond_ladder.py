"""Thermal conductivity workflow for Kitaev ladder using purification + MPO evolution.

Overall structure
-----------------
1) Build Hamiltonian MPO ``H`` and energy-current MPO ``J``.
2) For each temperature ``T`` (i.e. ``beta = 1/T``):
    a) Start from infinite-T purification ``|psi_inf>``.
    b) Imaginary-time evolve to ``|psi_beta>`` using ``exp(-tau H)`` with
        ``tau = beta/2`` (so that ``rho ~ exp(-beta H)``).
    c) Build two real-time branches:
        - Left branch: ``|L(t)> = exp(-iHt)|psi_beta>``
        - Right branch: ``|R(t)> = exp(-iHt)J|psi_beta>``
    d) Measure
        ``C_th(t) = Re[ <L(t)| J |R(t)> ] / L``
        which is exactly
        ``Re[ <psi_beta| exp(+iHt) J exp(-iHt) J |psi_beta> ] / L``.
3) Extract ``D_th(T)`` from long-time plateau (Eq. 4).
4) Build ``kappa_reg(omega, T)`` from finite-time Fourier transform of
    ``C_tilde(t) = C_th(t) - 2 T^2 D_th`` (Eq. 5).
5) Plot/Save:
    - contour of ``kappa_reg(omega, T)``,
    - ``D_th(T)``,
    - representative ``kappa_reg`` and full ``kappa`` using broadened Drude peak
      (Eq. 1, for visualization).

Notes on mixed imaginary-real evolution
---------------------------------------
- Imaginary time prepares thermal equilibrium.
- Real time is then applied on top of that equilibrium state to compute linear
  response correlators.
- The explicit left branch evolution above provides the required ``exp(+iHt)``
  on the bra side; it is not added by hand afterwards.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from models.model_Kladder import Kitaev_Ladder
from models.mpo_current import CurrentOperators
from tenpy.algorithms.purification import PurificationApplyMPO
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.purification_mps import PurificationMPS


# ==============================
# Hard-coded input parameters
# ==============================
LX = 11
J_K = -1.0
H_FIELD = 0.09
BC = "open"
BC_MPS = "finite"
ORDER = "default"

# Sweep temperatures T = 1 / beta
TEMP_LIST = [0.5]
DT_IMAG = 0.1
DT_REAL = 0.1
N_STEPS_REAL = 120

CHI_MAX = 70
SVD_MIN = 1.0e-8
TRUNC_CUT = 1.0e-8
APPROX = "II"
MAX_TRUNC_ERR_DEBUG = 1.0e-2

# current-density reference sites used to build the total current MPO
SITE_REFS = list(range(2, 2 * LX - 6, 4))

OUTFILE = "cond_ladder_data.txt"
OUTFIG = "cond_ladder_kappa.png"

# Frequency grid for kappa_reg(omega, T)
OMEGA_MAX = 8.0
N_OMEGA = 250

# Broadening for plotting the Drude delta-peak in Eq. (1)
DELTA_BROADENING_ETA = 0.15

# Fraction of tail used to estimate long-time asymptote in Eq. (4)
TAIL_FRACTION = 0.25

# Progress print cadence
PROGRESS_EVERY_IMAG_STEPS = 1
PROGRESS_EVERY_REAL_STEPS = 1


TRUNC_PARAMS = {
    "chi_max": CHI_MAX,
    "svd_min": SVD_MIN,
}
APPLY_OPTIONS = {
    "compression_method": "zip_up",
    "m_temp": 2,
    "trunc_weight": 0.5,
    "max_trunc_err": MAX_TRUNC_ERR_DEBUG,
    "trunc_params": TRUNC_PARAMS,
}


def _fmt_seconds(seconds):
    """Format elapsed seconds as ``HH:MM:SS`` for progress logs."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def overlap_mpo(bra, mpo, ket):
    """Return the scalar overlap ``<bra|MPO|ket>``.

    This is the basic measurement primitive used throughout the script, e.g. for
    evaluating current-current correlators.
    """
    env = MPOEnvironment(bra, mpo, ket)
    return env.full_contraction(0)


def build_models():
    """Construct the Kitaev ladder Hamiltonian and thermal current MPO models.

    Returns
    -------
    tuple
        ``(H_model, J_model)`` where:
        - ``H_model`` provides the Hamiltonian MPO for time evolution,
        - ``J_model`` provides the total energy-current MPO assembled from
          selected local current densities in ``SITE_REFS``.
    """
    model_params = dict(
        Lx=LX,
        order=ORDER,
        J_K=J_K,
        Fx=H_FIELD,
        Fy=H_FIELD,
        Fz=H_FIELD,
        bc=BC,
        bc_MPS=BC_MPS,
    )
    hamiltonian_model = Kitaev_Ladder(model_params)

    current_params = dict(model_params)
    current_params["siteRef"] = SITE_REFS
    current_model = CurrentOperators(current_params)

    return hamiltonian_model, current_model


def cool_to_target_beta(psi, h_mpo, beta_target):
    """Prepare a finite-temperature purification state by imaginary-time evolution.

    Purpose
    -------
    Starting from the infinite-temperature purification ``psi``, this function
    applies imaginary-time gates to obtain a state corresponding to inverse
    temperature ``beta_target``.

    Logic
    -----
    In purification, evolving the ket by ``exp(-tau H)`` corresponds to
    ``rho ~ exp(-2 tau H)``, so we target ``tau_target = beta_target / 2``.
    A second-order complex-time decomposition is used for each step.
    """
    # In purification, applying exp(-tau H) to |psi> corresponds to rho ~ exp(-2 tau H)
    tau_target = 0.5 * beta_target
    tau = 0.0
    imag_step = 0
    n_imag_steps = int(np.ceil(tau_target / DT_IMAG))
    t0 = time.time()

    Us_imag = [h_mpo.make_U(-d * DT_IMAG, APPROX) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(
        psi,
        Us_imag[0],
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )

    while tau < tau_target - 1.0e-12:
        for U in Us_imag:
            eng.init_env(U)
            eng.run()
        tau += DT_IMAG
        imag_step += 1
        if imag_step % PROGRESS_EVERY_IMAG_STEPS == 0 or imag_step == n_imag_steps:
            pct = 100.0 * imag_step / max(1, n_imag_steps)
            print(
                f"    imag-time: {imag_step:4d}/{n_imag_steps:4d} "
                f"({pct:5.1f}%)  tau={tau:.4f}/{tau_target:.4f}  "
                f"elapsed={_fmt_seconds(time.time() - t0)}"
            )

    return psi


def compute_correlator(H_model, J_model, beta_target):
    """Compute finite-temperature energy-current autocorrelation ``C_th(t)``.

    Purpose
    -------
    This function performs the mixed imaginary-real time pipeline used for Kubo
    transport:
    1) Cool from infinite-T to the target ``beta_target`` (imaginary time),
    2) Build the right branch ``|R(0)> = J|psi_beta>``,
    3) Build the left branch ``|L(0)> = |psi_beta>``,
    4) Real-time evolve both branches with ``exp(-iHt)``,
    5) Measure ``<L(t)|J|R(t)>``.

    This equals ``<psi_beta|exp(+iHt) J exp(-iHt) J|psi_beta>`` 

    Returns
    -------
    times : ndarray
        Real-time grid.
    c_th : ndarray
        ``Re< J(t)J(0) >/L`` used in Eq. (4) and Eq. (5).
    """
    psi_beta = PurificationMPS.from_infiniteT(H_model.lat.mps_sites(), bc=BC_MPS)
    psi_beta = cool_to_target_beta(psi_beta, H_model.H_MPO, beta_target)

    left = psi_beta.copy()
    right = psi_beta.copy()
    eng_apply_j = PurificationApplyMPO(
        right,
        J_model.H_MPO,
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )
    eng_apply_j.run()

    # this is the same as ExpMPOEvolution in https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mpo.MPO.html#tenpy.networks.mpo.MPO.make_U 
    U_real = H_model.H_MPO.make_U(-1.0j * DT_REAL, APPROX)
    
    eng_left = PurificationApplyMPO(
        left,
        U_real,
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )
    eng_right = PurificationApplyMPO(
        right,
        U_real,
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )

    times = [0.0]
    corr = [overlap_mpo(left, J_model.H_MPO, right)]
    t0 = time.time()

    for step in range(1, N_STEPS_REAL + 1):
        eng_left.init_env(U_real)
        eng_left.run()
        eng_right.init_env(U_real)
        eng_right.run()
        times.append(step * DT_REAL)
        corr.append(overlap_mpo(left, J_model.H_MPO, right))
        if step % PROGRESS_EVERY_REAL_STEPS == 0 or step == N_STEPS_REAL:
            pct = 100.0 * step / max(1, N_STEPS_REAL)
            print(
                f"    real-time: {step:4d}/{N_STEPS_REAL:4d} ({pct:5.1f}%)  "
                f"t={times[-1]:.4f}  elapsed={_fmt_seconds(time.time() - t0)}"
            )

    times = np.array(times, dtype=float)
    corr = np.array(corr, dtype=complex)
    # C_th(t) in the convention used in the paper: Re <J(t)J> / L
    c_th = corr.real / H_model.lat.N_sites
    return times, c_th


def estimate_drude_weight(c_th, temperature):
    """Estimate thermal Drude weight ``D_th(T)`` from long-time correlator tail.

    Implements Eq. (4) in finite-time form by averaging the last fraction of
    ``C_th(t)`` values as a proxy for ``t -> infinity`` and then dividing by
    ``2*T^2``.
    """
    n_tail = max(3, int(TAIL_FRACTION * len(c_th)))
    c_inf_est = np.mean(c_th[-n_tail:])
    d_th = c_inf_est / (2.0 * temperature * temperature)
    return d_th


def kappa_regular_from_ctilde(times, c_tilde, temperature, omegas):
    """Compute ``kappa_reg(omega, T)`` from the connected correlator ``C_tilde``.

    Purpose
    -------
    Evaluate Eq. (5):
    ``kappa_reg(omega) = ((1-exp(-omega/T))/(omega*T)) * Re \\int_0^inf dt e^{iwt} C_tilde(t)``.

    Numerical details
    -----------------
    - Uses finite integration range set by simulated times,
    - Applies a Hann window to reduce ringing from time truncation,
    - Uses the smooth ``omega->0`` limit ``1/T^2`` for the prefactor.
    """
    # Eq. (5): kappa_reg(omega)
    # kappa_reg(omega)=((1-exp(-omega/T))/(omega*T)) * Re int_0^inf dt e^{i omega t} C_tilde(t)
    # finite-time implementation with Hann window
    window = np.hanning(len(times))
    c_win = c_tilde * window

    kappa_reg = np.zeros_like(omegas)
    for idx, omega in enumerate(omegas):
        kernel = np.exp(1.0j * omega * times)
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(c_win * kernel, times).real
        else:
            integral = np.trapz(c_win * kernel, times).real
        if np.isclose(omega, 0.0):
            prefactor = 1.0 / (temperature * temperature)
        else:
            prefactor = (1.0 - np.exp(-omega / temperature)) / (omega * temperature)
        kappa_reg[idx] = prefactor * integral
    return kappa_reg


def broadened_delta(omegas, eta):
    """Return Lorentzian approximation to ``delta(omega)`` for plotting.

    This is only for visualization of Eq. (1):
    ``2*pi*D_th*delta(omega) + kappa_reg(omega)``.
    """
    # Lorentzian representation of delta(omega): delta_eta(omega)=eta/[pi(omega^2+eta^2)]
    return eta / (np.pi * (omegas * omegas + eta * eta))


def main():
    """Run temperature sweep and generate conductivity data/figures.

    What this orchestrates
    ----------------------
    For each temperature in ``TEMP_LIST``:
    - Build ``C_th(t)`` via mixed imaginary-real evolution,
    - Extract ``D_th(T)`` from Eq. (4),
    - Build ``kappa_reg(omega, T)`` from Eq. (5).

    Outputs
    -------
    - ``OUTFILE``: summary table vs temperature,
    - ``OUTFIG``: three-panel plot with contour ``kappa_reg(omega, T)``,
      ``D_th(T)``, and a regular/full conductivity comparison using broadened
      Drude peak from Eq. (1).
    """
    H_model, J_model = build_models()

    omegas = np.linspace(0.0, OMEGA_MAX, N_OMEGA)
    temperatures = np.array(TEMP_LIST, dtype=float)
    betas = 1.0 / temperatures

    d_th_all = []
    kappa_reg_all = []
    kappa_reg_dc_all = []
    times_ref = None
    sweep_t0 = time.time()
    n_temp = len(temperatures)

    for idx, (temp, beta) in enumerate(zip(temperatures, betas), start=1):
        temp_t0 = time.time()
        print(f"[{idx}/{n_temp}] T={temp:.4f}, beta={beta:.4f}")
        times, c_th = compute_correlator(H_model, J_model, beta)
        if times_ref is None:
            times_ref = times

        d_th = estimate_drude_weight(c_th, temp)  # Eq. (4)
        c_tilde = c_th - 2.0 * temp * temp * d_th
        kappa_reg = kappa_regular_from_ctilde(times, c_tilde, temp, omegas)  # Eq. (5)

        d_th_all.append(d_th)
        kappa_reg_all.append(kappa_reg)
        kappa_reg_dc_all.append(kappa_reg[0])
        done = idx
        elapsed = time.time() - sweep_t0
        avg = elapsed / done
        eta = avg * (n_temp - done)
        print(
            f"  done T={temp:.4f} in {_fmt_seconds(time.time() - temp_t0)} | "
            f"sweep {done}/{n_temp} ({100.0*done/n_temp:5.1f}%) | "
            f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}"
        )

    d_th_all = np.array(d_th_all)
    kappa_reg_all = np.array(kappa_reg_all)
    kappa_reg_dc_all = np.array(kappa_reg_dc_all)

    # Eq. (1): Re kappa(omega) = 2 pi D_th delta(omega) + kappa_reg(omega)
    # For plotting only, represent delta by a Lorentzian with width eta.
    t_pick_idx = len(temperatures) // 2
    t_pick = temperatures[t_pick_idx]
    d_pick = d_th_all[t_pick_idx]
    kappa_reg_pick = kappa_reg_all[t_pick_idx]
    kappa_full_pick = kappa_reg_pick + 2.0 * np.pi * d_pick * broadened_delta(omegas, DELTA_BROADENING_ETA)

    # A finite visual proxy for "full conductivity on top of regular part" at omega=0
    kappa_full_dc_visual = kappa_reg_dc_all + 2.0 * d_th_all / DELTA_BROADENING_ETA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: contour kappa_reg(omega, T), or line if only one temperature
    if kappa_reg_all.shape[0] >= 2 and kappa_reg_all.shape[1] >= 2:
        cf = axes[0].contourf(omegas, temperatures, kappa_reg_all, levels=40, cmap="viridis")
        axes[0].set_xlabel(r"$\omega$")
        axes[0].set_ylabel(r"$T$")
        axes[0].set_title(r"$\kappa^{\mathrm{reg}}(\omega, T)$")
        fig.colorbar(cf, ax=axes[0], label=r"$\kappa^{\mathrm{reg}}$")
    else:
        axes[0].plot(omegas, kappa_reg_all[0], lw=2)
        axes[0].set_xlabel(r"$\omega$")
        axes[0].set_ylabel(r"$\kappa^{\mathrm{reg}}(\omega)$")
        axes[0].set_title(rf"$\kappa^{{\mathrm{{reg}}}}(\omega)$ at $T={temperatures[0]:.2f}$")
        axes[0].grid(alpha=0.3)

    # Panel 2: thermal Drude weight Eq. (4)
    axes[1].plot(temperatures, d_th_all, "o-", lw=2)
    axes[1].set_xlabel(r"$T$")
    axes[1].set_ylabel(r"$D_{\mathrm{th}}(T)$")
    axes[1].set_title(r"Drude weight (Eq. 4)")
    axes[1].grid(alpha=0.3)

    # Panel 3: regular vs full conductivity (Eq. 1) at one representative T
    axes[2].plot(omegas, kappa_reg_pick, lw=2, label=rf"$\kappa^{{\rm reg}}(\omega)$ at $T={t_pick:.2f}$")
    axes[2].plot(
        omegas,
        kappa_full_pick,
        "--",
        lw=2,
        label=rf"$\kappa(\omega)=2\pi D_{{\rm th}}\delta_\eta+\kappa^{{\rm reg}}$ ($\eta={DELTA_BROADENING_ETA:.2f}$)",
    )
    axes[2].set_xlabel(r"$\omega$")
    axes[2].set_ylabel(r"$\kappa(\omega)$")
    axes[2].set_title(r"Regular + full conductivity (Eq. 1)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTFIG, dpi=220, bbox_inches="tight")

    data_header = (
        "Kitaev ladder thermal conductivity from current-current correlators\n"
        f"Lx={LX}, N_sites={H_model.lat.N_sites}, J_K={J_K}, h={H_FIELD}, bc={BC}, order={ORDER}\n"
        f"dt_imag={DT_IMAG}, dt_real={DT_REAL}, N_steps_real={N_STEPS_REAL}, chi_max={CHI_MAX}\n"
        f"omega_max={OMEGA_MAX}, N_omega={N_OMEGA}, eta={DELTA_BROADENING_ETA}\n"
        f"site_refs={SITE_REFS}\n"
        "Columns: T | D_th(Eq4) | kappa_reg(omega=0) | kappa_full_dc_visual"
    )
    output_table = np.column_stack([temperatures, d_th_all, kappa_reg_dc_all, kappa_full_dc_visual])
    np.savetxt(OUTFILE, output_table, header=data_header, comments="# ")

    print("Saved conductivity summary to", OUTFILE)
    print("Saved figure to", OUTFIG)
    print("Note: PurificationTEBD is not used because this ladder ordering has non-nearest-neighbor couplings in MPS order.")


if __name__ == "__main__":
    main()
