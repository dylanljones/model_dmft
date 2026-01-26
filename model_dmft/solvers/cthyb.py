# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-06

from typing import Tuple, Union

import numpy as np
import triqs.operators as ops
import triqs_cthyb
from triqs.gf import BlockGf, inverse, iOmega_n
from triqs.utility import mpi
from triqs_cthyb.tail_fit import tail_fit

from ..input import CthybSolverParams, InputParameters
from ..stabilize import apply_legendre_filter, crm_solve_dyson, legendre_fit, pick_nl_opt, pick_wmax_opt, truncate_g_l
from ..utility import rebin_gf_tau, report


def solve_cthyb(params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf) -> triqs_cthyb.Solver:
    up, dn = params.spin_names
    solver_params: CthybSolverParams = params.solver_params
    # gf_struct = params.gf_struct
    mu = params.mu

    report("Initializing CTHYB solver...")

    # Local Hamiltonian and interaction term
    eps = e_onsite  #  - mu
    # Important: shift interaction to have zero Hartree at half-filling
    h_int = u * (ops.n(up, 0) - 0.5) * (ops.n(dn, 0) - 0.5)

    # h_loc0 = eps[0] * ops.n(up, 0) + eps[1] * ops.n(dn, 0)
    # h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)

    # Initialize delta interface
    g0_iw = delta.copy()
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n + mu - delta[name] - eps[i])

    solve_kwargs = {
        "n_warmup_cycles": solver_params.n_warmup_cycles,
        "n_cycles": solver_params.n_cycles,
        "length_cycle": solver_params.length_cycle,
        "measure_G_l": solver_params.measure_g_l or False,
        # Used for calculating moments
        "measure_density_matrix": True,
        "use_norm_as_weight": True,
    }

    # Ensure different random seed on each MPI rank, but reproducible
    seed_base = solver_params.random_seed if solver_params.random_seed is not None else 34788
    solve_kwargs["random_seed"] = seed_base + 928374 * mpi.rank  # random seed on each core

    # Set random name if provided
    if solver_params.random_name:
        solve_kwargs["random_name"] = solver_params.random_name  # noqa

    # Initialize solver
    solver = triqs_cthyb.Solver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        n_iw=params.n_iw,
        n_tau=solver_params.n_tau,
        # delta_interface=True,
        n_l=solver_params.n_l or solver_params.nl_max or 100,
    )

    # Set hybridization function (imaginary time)
    # for spin, delt in delta:
    #     tail, err = fit_hermitian_tail(delt)
    #     delt << delt - tail[0]  # noqa
    # solver.Delta_tau << Fourier(delta)  # type: ignore

    solver.G0_iw << g0_iw  # noqa
    mpi.barrier()

    # Solve impurity problem
    report("Solving impurity...")
    solver.solve(h_int=h_int, **solve_kwargs)
    report("Done!")
    report("")

    return solver


def postprocess_cthyb(
    params: InputParameters, solver: triqs_cthyb.Solver, u: float
) -> Tuple[Union[BlockGf, None], Union[BlockGf, None], Union[BlockGf, None]]:
    """Postprocess the CTHYB solver output.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    solver : triqs_cthyb.Solver
        The CTHYB solver object.
    u : float
        The interaction strength.

    Returns
    -------
    sigma_iw : BlockGf
        The self-energy obtained by postprocessing the CTHYB solver output.
    g_l : BlockGf
        The Legendre Green's function used in the postprocessing. Only returned if Legendre fitting
         is used.
    g_tau_rebinned : BlockGf
        The re-binned imaginary time Green's function.
    """
    solver_params = params.solver_params
    # Output quantities
    sigma_iw = None
    g_l = None
    g_tau_rebinned = None

    if solver_params.rebin_tau:
        report("Re-binning G(τ)...")
        g_tau_rebinned = rebin_gf_tau(solver.G_tau, solver_params.rebin_tau)

    if solver_params.measure_g_l:
        g_l = solver.G_l

    if solver_params.tail_fit:
        report("Performing tail fit of of Σ(iw)...")
        fit_max_moment = params.solver_params.fit_max_moment

        try:
            sigma_moments = solver.Sigma_moments
        except AttributeError:
            sigma_moments = None
            report("")
            report("WARNING: No moments found in solver. Are you using the latest version?")
            report("")

        sigma_fitted = solver.Sigma_iw.copy()
        tail_fit(
            sigma_fitted,
            fit_min_n=params.solver_params.fit_min_n,
            fit_max_n=params.solver_params.fit_max_n,
            fit_min_w=params.solver_params.fit_min_w,
            fit_max_w=params.solver_params.fit_max_w,
            fit_max_moment=fit_max_moment,
            fit_known_moments=sigma_moments,
        )
        sigma_iw = sigma_fitted

    elif solver_params.legendre_fit:
        try:
            sigma_moments = solver.Sigma_moments
        except AttributeError:
            raise ValueError("No moments found in solver. Are you using the latest version?")

        if g_l is None:
            # Compute Legendre Gf by filtering binned imaginary time Green's function
            report("Applying Legendre filter...")
            order = solver_params.n_l or solver_params.nl_max or 100
            g_l = apply_legendre_filter(solver.G_tau, order=order)

        if solver_params.n_l is None:
            report("\nOptimizing n_l for Legendre fit...\n\n")
            res = pick_nl_opt(
                g_l,
                solver.G0_iw,
                solver.G_iw,
                sigma_moments,
                nl_step=params.solver_params.nl_step,
                smooth=8,
                smooth_err=30,
                iw_stop=solver_params.legendre_iw_stop or 50,
                tail_frac=0.8,
                idx_step=1,
                tol=solver_params.legendre_tol or 1e-2,
                q=solver_params.legendre_q or None,
                consec=solver_params.legendre_consec or 1,
            )
            if not res.success:
                n_l_default = 15
                report("================================================================")
                report(f"WARNING: Did not find optimal n_l! Using n_l={n_l_default}.")
                report("================================================================")
                g_l_trunc = truncate_g_l(g_l, n_l_default)
                g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(solver.G0_iw, solver.G_iw, solver.G_tau, g_l_trunc)
            else:
                n_l = res.nl
                g_l = truncate_g_l(g_l, n_l)
                report(f"\nOptimal w_max found: {n_l}")
                sigma_iw_l = res.sigma_opt
        else:
            # Fit the Green's functions and self energy using the Legendre Green's function
            report("Performing Legendre fit...")
            g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(solver.G0_iw, solver.G_iw, solver.G_tau, g_l)

        sigma_iw = sigma_iw_l

    elif solver_params.crm_dyson:
        report("Solving Dyson equation via constrained minimization problem...")
        try:
            sigma_moments = solver.Sigma_moments
        except AttributeError:
            raise ValueError("No moments found in solver. Are you using the latest version?")

        g_tau = solver.G_tau if g_tau_rebinned is None else g_tau_rebinned

        if solver_params.crm_wmax is None:
            report("\nOptimizing wmax for CRM Dyson solver...\n\n")
            # Optimize wmax based on Matsubara frequencies
            d = params.half_bandwidth if params.half_bandwidth is not None else params.t
            res = pick_wmax_opt(
                g_tau,
                solver.G0_iw,
                solver.G_iw,
                sigma_moments,
                start=solver_params.crm_wmax_start or 1.0 * d,
                stop=solver_params.crm_wmax_end or 5.0 * d,
                step=solver_params.crm_wmax_step or 0.1 * d,
                smooth=8,
                smooth_err=30,
                iw_stop=solver_params.crm_iw_stop or 50,
                tail_frac=0.5,
                idx_step=1,
                tol=solver_params.crm_tol or 1e-2,
                q=solver_params.crm_q or None,
                consec=solver_params.crm_consec or 1,
            )
            if not res.success:
                wmax_center = (solver_params.crm_wmax_start + solver_params.crm_wmax_end) / 2.0
                report("================================================================")
                report(f"WARNING: Did not find optimal w_max! Using w_max={wmax_center}.")
                report("================================================================")
                sigma_iw_crm = crm_solve_dyson(g_tau, solver.G0_iw, sigma_moments, wmax_center, solver_params.crm_eps)
            else:
                wmax_opt = res.wmax
                report(f"\nOptimal w_max found: {wmax_opt}")
                sigma_iw_crm = res.sigma_opt
        else:
            sigma_iw_crm = crm_solve_dyson(
                g_tau, solver.G0_iw, sigma_moments, solver_params.crm_wmax, solver_params.crm_eps
            )

        sigma_iw = sigma_iw_crm

    return sigma_iw, g_l, g_tau_rebinned


def postprocess_cthyb_old(
    params: InputParameters, solver: triqs_cthyb.Solver, u: float
) -> Tuple[Union[BlockGf, None], Union[BlockGf, None], Union[BlockGf, None]]:
    """Postprocess the CTHYB solver output.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    solver : triqs_cthyb.Solver
        The CTHYB solver object.
    u : float
        The interaction strength.

    Returns
    -------
    sigma_iw : BlockGf
        The self-energy obtained by postprocessing the CTHYB solver output.
    g_l : BlockGf
        The Legendre Green's function used in the postprocessing. Only returned if Legendre fitting
         is used.
    g_tau_rebinned : BlockGf
        The re-binned imaginary time Green's function.
    """
    solver_params = params.solver_params
    # Output quantities
    sigma_iw = None
    g_l = None
    g_tau_rebinned = None

    if solver_params.rebin_tau:
        report("Re-binning G(τ)...")
        g_tau_rebinned = rebin_gf_tau(solver.G_tau, solver_params.rebin_tau)

    if solver_params.measure_g_l:
        g_l = solver.G_l

    if solver_params.legendre_fit:
        if g_l is None:
            # Compute Legendre Gf by filtering binned imaginary time Green's function
            report("Applying Legendre filter...")
            g_l = apply_legendre_filter(solver.G_tau, order=solver_params.n_l)

        # Fit the Green's functions and self energy using the Legendre Green's function
        report("Performing Legendre fit...")
        g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(solver.G0_iw, solver.G_iw, solver.G_tau, g_l)

        sigma_iw = sigma_iw_l

    elif solver_params.tail_fit:
        report("Performing tail fit of of Σ(iw)...")
        fit_max_moment = params.solver_params.fit_max_moment

        try:
            sigma_moments = solver.Sigma_moments
        except AttributeError:
            sigma_moments = None
            report("")
            report("WARNING: No moments found in solver. Are you using the latest version?")
            report("")

        sigma_fitted = solver.Sigma_iw.copy()
        tail_fit(
            sigma_fitted,
            fit_min_n=params.solver_params.fit_min_n,
            fit_max_n=params.solver_params.fit_max_n,
            fit_min_w=params.solver_params.fit_min_w,
            fit_max_w=params.solver_params.fit_max_w,
            fit_max_moment=fit_max_moment,
            fit_known_moments=sigma_moments,
        )
        sigma_iw = sigma_fitted

    elif solver_params.crm_dyson:
        report("Solving Dyson equation via constrained minimization problem...")
        try:
            sigma_moments = solver.Sigma_moments
        except AttributeError:
            raise ValueError("No moments found in solver. Are you using the latest version?")

        g_tau = solver.G_tau if g_tau_rebinned is None else g_tau_rebinned

        sigma_iw_crm = crm_solve_dyson(
            g_tau, solver.G0_iw, sigma_moments, solver_params.crm_wmax, solver_params.crm_eps
        )

        sigma_iw = sigma_iw_crm

    # if solver_params.correct_hartree:
    #     # Correct hartree shift
    #     report("Correcting Hartree shift...")
    #     densities = dict()
    #     for spin, g in solver.G_iw:
    #         # Compute density
    #         dens = g.density()
    #         if np.any(dens.imag > 1e-10):
    #             report("Warning: density is not real")
    #         densities[spin] = dens.real
    #
    #     up, dn = params.spin_names
    #     correction_up = densities[dn] * u
    #     correction_dn = densities[up] * u
    #     sigma_iw[up] += correction_up
    #     sigma_iw[dn] += correction_dn
    #     report(f"Corrected Hartree shift: {correction_up} (up), {correction_dn} (dn)")

    return sigma_iw, g_l, g_tau_rebinned
