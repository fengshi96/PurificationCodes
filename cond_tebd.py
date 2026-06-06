"""Thermal-current correlator with supersite grouping + PurificationTEBD disentangler.

Why supersites?
---------------
For the ladder in default MPS ordering, couplings are not nearest-neighbor, so
`PurificationTEBD` cannot be used directly. We group 2 physical sites into one
supersite to reduce interaction range in the grouped chain and then construct
`H_bond` from the grouped MPO.

Workflow implemented here
-------------------------
1) Build `H` (Kitaev ladder MPO) and current MPO `J`.
2) Group both MPOs with `group_sites(n=2)`.
3) Set `H_model.H_bond = H_model.calc_H_bond_from_MPO()` for TEBD updates.
4) Build purification state on grouped sites at infinite T.
5) Imaginary-time cooling to target beta using `PurificationApplyMPO`.
6) Build two branches for correlator:
   - left:  |L(0)> = |psi_beta>
   - right: |R(0)> = J |psi_beta>
7) Real-time evolve BOTH branches with `PurificationTEBD` and `disentangle` enabled.
8) Measure C_th(t) = Re <L(t)|J|R(t)> / (L_grouped * Z_beta).

Notes
-----
- The finite-T disentangler is active in the real-time TEBD part.
- Imaginary-time cooling here is done via MPO-apply for robustness with this model.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from models.model_Kladder import Kitaev_Ladder
from models.mpo_current import CurrentOperators
from tenpy.algorithms.purification import PurificationApplyMPO, PurificationTEBD
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.purification_mps import PurificationMPS


# -----------------------------
# Hard-coded debug parameters
# -----------------------------
LX = 11
J_K = -1.0
H_FIELD = 0.09
BC = "open"
BC_MPS = "finite"
ORDER = "default"

# Use TEMP_LIST for direct comparison with cond_ladder.py
TEMP_LIST = [0.5]
DT_IMAG = 0.1
DT_REAL = 0.1
N_STEPS_REAL = 120

GROUP_N = 2

CHI_MAX = 70
SVD_MIN = 1e-8
APPROX = "II"
MAX_TRUNC_ERR_DEBUG = 5e-2

SITE_REFS = list(range(2, 2 * LX - 6, 4))
OUTFILE = "cond_tebd_data.txt"
OUTFIG = "cond_tebd_kappa.png"

OMEGA_MAX = 8.0
N_OMEGA = 250
DELTA_BROADENING_ETA = 0.15
TAIL_FRACTION = 0.25

TRUNC_PARAMS = {"chi_max": CHI_MAX, "svd_min": SVD_MIN}


def overlap_mpo(bra, mpo, ket):
    """Return scalar overlap <bra|MPO|ket>."""
    return MPOEnvironment(bra, mpo, ket).full_contraction(0)


def build_grouped_models():
    """Build H and J models, then group MPO sites into supersites."""
    params = dict(
        Lx=LX,
        order=ORDER,
        J_K=J_K,
        Fx=H_FIELD,
        Fy=H_FIELD,
        Fz=H_FIELD,
        bc=BC,
        bc_MPS=BC_MPS,
    )

    h_model = Kitaev_Ladder(params)
    j_params = dict(params)
    j_params["siteRef"] = SITE_REFS
    j_model = CurrentOperators(j_params)

    h_model.H_MPO.group_sites(n=GROUP_N)
    j_model.H_MPO.group_sites(n=GROUP_N)

    # Required by PurificationTEBD.
    h_model.H_bond = h_model.calc_H_bond_from_MPO()

    return h_model, j_model


def cool_to_beta_apply_mpo(psi, h_mpo, beta_target):
    """Imaginary-time cooling with MPO-apply: tau = beta/2."""
    tau_target = 0.5 * beta_target
    n_steps = int(np.ceil(tau_target / DT_IMAG))
    tau = 0.0

    Us = [h_mpo.make_U(-d * DT_IMAG, APPROX) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(
        psi,
        Us[0],
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )

    t0 = time.time()
    for k in range(1, n_steps + 1):
        for U in Us:
            eng.init_env(U)
            eng.run()
        tau += DT_IMAG
        print(f"imag-step {k:4d}/{n_steps:4d} tau={tau:.4f}/{tau_target:.4f} elapsed={time.time()-t0:.1f}s")

    return psi


def compute_correlator_tebd_disentangler(h_model, j_model, beta_target):
    """Compute C_th(t) using TEBD real-time with finite-T disentangler enabled."""
    psi_beta = PurificationMPS.from_infiniteT(h_model.H_MPO.sites, bc=BC_MPS)
    psi_beta = cool_to_beta_apply_mpo(psi_beta, h_model.H_MPO, beta_target)

    z_beta = psi_beta.overlap(psi_beta)

    left = psi_beta.copy()
    right = psi_beta.copy()

    # right <- J|psi_beta>
    eng_apply_j = PurificationApplyMPO(
        right,
        j_model.H_MPO,
        {"trunc_params": TRUNC_PARAMS, "max_trunc_err": MAX_TRUNC_ERR_DEBUG},
    )
    eng_apply_j.run()

    # TEBD engines with disentangler enabled.
    tebd_opts = {
        "trunc_params": TRUNC_PARAMS,
        "disentangle": "last-renyi",
        "max_trunc_err": MAX_TRUNC_ERR_DEBUG,
    }
    eng_left = PurificationTEBD(left, h_model, tebd_opts)
    eng_right = PurificationTEBD(right, h_model, tebd_opts)

    times = [0.0]
    corr = [overlap_mpo(left, j_model.H_MPO, right) / z_beta]
    s_ent = [np.max(left.entanglement_entropy())]

    t0 = time.time()
    for step in range(1, N_STEPS_REAL + 1):
        eng_left.run_evolution(1, DT_REAL)
        eng_right.run_evolution(1, DT_REAL)
        times.append(step * DT_REAL)
        corr.append(overlap_mpo(left, j_model.H_MPO, right) / z_beta)
        s_ent.append(np.max(left.entanglement_entropy()))
        print(f"real-step {step:4d}/{N_STEPS_REAL:4d} t={times[-1]:.4f} elapsed={time.time()-t0:.1f}s")

    times = np.array(times, dtype=float)
    corr = np.array(corr, dtype=complex)
    s_ent = np.array(s_ent, dtype=float)

    # grouped chain length after supersite grouping
    l_grouped = h_model.H_MPO.L
    c_th = corr.real / l_grouped
    return times, c_th, s_ent


def estimate_drude_weight(c_th, temperature):
    """Estimate D_th from the late-time average of C_th(t), Eq. (4)."""
    n_tail = max(3, int(TAIL_FRACTION * len(c_th)))
    c_inf_est = np.mean(c_th[-n_tail:])
    return c_inf_est / (2.0 * temperature * temperature)


def kappa_regular_from_ctilde(times, c_tilde, temperature, omegas):
    """Compute kappa_reg(omega, T) from Eq. (5) using finite-time integration."""
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
    """Lorentzian approximation to delta(omega) for plotting Eq. (1)."""
    return eta / (np.pi * (omegas * omegas + eta * eta))


def main():
    """Run grouped-supersite TEBD-disentangler transport workflow and save output."""
    h_model, j_model = build_grouped_models()
    omegas = np.linspace(0.0, OMEGA_MAX, N_OMEGA)
    temperatures = np.array(TEMP_LIST, dtype=float)
    betas = 1.0 / temperatures

    d_th_all = []
    kappa_reg_all = []
    kappa_reg_dc_all = []
    c_th_all = []
    s_ent_all = []
    times_ref = None

    for idx, (temp, beta) in enumerate(zip(temperatures, betas), start=1):
        print(f"[{idx}/{len(temperatures)}] T={temp:.4f}, beta={beta:.4f}")
        times, c_th, s_ent = compute_correlator_tebd_disentangler(h_model, j_model, beta)
        if times_ref is None:
            times_ref = times
        c_th_all.append(c_th)
        s_ent_all.append(s_ent)
        d_th = estimate_drude_weight(c_th, temp)
        c_tilde = c_th - 2.0 * temp * temp * d_th
        kappa_reg = kappa_regular_from_ctilde(times, c_tilde, temp, omegas)
        d_th_all.append(d_th)
        kappa_reg_all.append(kappa_reg)
        kappa_reg_dc_all.append(kappa_reg[0])

    d_th_all = np.array(d_th_all)
    kappa_reg_all = np.array(kappa_reg_all)
    kappa_reg_dc_all = np.array(kappa_reg_dc_all)
    c_th_all = np.array(c_th_all)
    s_ent_all = np.array(s_ent_all)

    t_pick_idx = len(temperatures) // 2
    t_pick = temperatures[t_pick_idx]
    d_pick = d_th_all[t_pick_idx]
    kappa_reg_pick = kappa_reg_all[t_pick_idx]
    kappa_full_pick = kappa_reg_pick + 2.0 * np.pi * d_pick * broadened_delta(omegas, DELTA_BROADENING_ETA)

    kappa_full_dc_visual = kappa_reg_dc_all + 2.0 * d_th_all / DELTA_BROADENING_ETA

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

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

    axes[1].plot(temperatures, d_th_all, "o-", lw=2)
    axes[1].set_xlabel(r"$T$")
    axes[1].set_ylabel(r"$D_{\mathrm{th}}(T)$")
    axes[1].set_title(r"Drude weight (Eq. 4)")
    axes[1].grid(alpha=0.3)

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

    # Panel 4: C_th(t) at representative temperature
    c_th_pick = c_th_all[t_pick_idx]
    axes[3].plot(times_ref, c_th_pick, lw=2, label=rf"$C_{{\rm th}}(t)$ at $T={t_pick:.2f}$")
    axes[3].set_xlabel(r"$t$")
    axes[3].set_ylabel(r"$C_{\mathrm{th}}(t)$")
    axes[3].set_title(r"Current correlator vs time")
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=8)

    # Panel 5: entanglement growth vs time (left branch)
    s_ent_pick = s_ent_all[t_pick_idx]
    axes[4].plot(times_ref, s_ent_pick, lw=2, label=rf"$S_{{\rm ent}}(t)$ at $T={t_pick:.2f}$")
    axes[4].set_xlabel(r"$t$")
    axes[4].set_ylabel(r"$S_{\mathrm{ent}}(t)$")
    axes[4].set_title(r"Entanglement growth")
    axes[4].grid(alpha=0.3)
    axes[4].legend(fontsize=8)

    # Panel 6 unused
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(OUTFIG, dpi=220, bbox_inches="tight")

    header = (
        "Grouped-supersite TEBD + disentangler thermal conductivity summary\n"
        f"Lx={LX}, grouped_n={GROUP_N}, L_grouped={h_model.H_MPO.L}\n"
        f"dt_imag={DT_IMAG}, dt_real={DT_REAL}, n_steps_real={N_STEPS_REAL}\n"
        f"chi_max={CHI_MAX}, disentangle=last-renyi, omega_max={OMEGA_MAX}, N_omega={N_OMEGA}\n"
        "Columns: T | D_th(Eq4) | kappa_reg(omega=0) | kappa_full_dc_visual"
    )
    output_table = np.column_stack([temperatures, d_th_all, kappa_reg_dc_all, kappa_full_dc_visual])
    np.savetxt(OUTFILE, output_table, header=header, comments="# ")

    print("Saved", OUTFILE)
    print("Saved", OUTFIG)


if __name__ == "__main__":
    main()
