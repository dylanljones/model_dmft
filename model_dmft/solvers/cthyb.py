# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-06

from typing import Tuple, Union

import numpy as np
import triqs.operators as ops
import triqs_cthyb
from triqs.gf import (
    BlockGf,
    Gf,
    MeshDLRImFreq,
    fit_gf_dlr,
    inverse,
    iOmega_n,
    make_gf_imfreq,
    make_hermitian,
)
from triqs.gf.dlr_crm_dyson_solver import minimize_dyson
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi
from triqs_cthyb.tail_fit import tail_fit

from ..input import CthybSolverParams, InputParameters
from ..legendre import apply_legendre_filter
from ..utility import report


def solve_cthyb(params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf) -> triqs_cthyb.Solver:
    up, dn = params.spin_names
    solver_params: CthybSolverParams = params.solver_params
    gf_struct = params.gf_struct
    mu = params.mu

    report("Initializing CTHYB solver...")

    # Local Hamiltonian and interaction term
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)

    # Initialize delta interface
    g0_iw = delta.copy()
    h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n + mu - delta[name] - h_loc0_mat[i])  # maybe +mu missing?

    solve_kwargs = {
        "n_warmup_cycles": solver_params.n_warmup_cycles,
        "n_cycles": solver_params.n_cycles,
        "length_cycle": solver_params.length_cycle,
        "measure_G_l": solver_params.measure_g_l,
    }
    if solver_params.density_matrix:
        solve_kwargs["measure_density_matrix"] = True
        solve_kwargs["use_norm_as_weight"] = True

    if solver_params.tail_fit or solver_params.crm_dyson:
        # Used for calculating moments
        solve_kwargs["measure_density_matrix"] = True
        solve_kwargs["use_norm_as_weight"] = True

    seed_base = solver_params.random_seed if solver_params.random_seed is not None else 34788
    solve_kwargs["random_seed"] = seed_base + 928374 * mpi.rank  # random seed on each core
    if solver_params.random_name:
        solve_kwargs["random_name"] = solver_params.random_name

    # Initialize solver
    solver = triqs_cthyb.Solver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        n_iw=params.n_iw,
        n_tau=solver_params.n_tau,
        # delta_interface=True,
        n_l=solver_params.n_l,
    )
    # Set hybridization function (imaginary time)
    # solver.Delta_tau << Fourier(delta)  # type: ignore
    solver.G0_iw << g0_iw  # noqa
    mpi.barrier()

    # Solve impurity problem
    report("Solving impurity...")
    solver.solve(h_loc0=h_loc0, h_int=h_int, **solve_kwargs)
    report("Done!")
    report("")

    return solver


def legendre_fit(g0_iw: BlockGf, g_iw: BlockGf, g_tau: BlockGf, g_l: BlockGf) -> Tuple[BlockGf, BlockGf, BlockGf]:
    """Fit the Green's functions and self energy using the Legendre Green's function."""
    g_iw_l = g_iw.copy()
    g_tau_l = g_tau.copy()
    for name, g in g_l:
        g.enforce_discontinuity(np.identity(g.target_shape[0]))
        g_iw_l[name].set_from_legendre(g)
        g_tau_l[name].set_from_legendre(g)

    g_iw_l << make_hermitian(g_iw_l)
    sigma_iw_l = inverse(g0_iw) - inverse(g_iw_l)
    return g_iw_l, g_tau_l, sigma_iw_l


def crm_solve_dyson(
    g_tau: BlockGf, g0_iw: BlockGf, sigma_moments: dict[str, np.ndarray], w_max: float, eps: float
) -> BlockGf:
    """Solve the Dyson equation via a constrained minimization problem (CRM).

    Parameters
    ----------
    g_tau : imaginary time BlockGf
        The imaginary time Green's function measured by the impurity solver.
    g0_iw : imaginary frequency BlockGf
        The non-interacting Green's function.
    sigma_moments : dict[str, np.ndarray]
        The moments of the self-energy to be used in the CRM solver.
    w_max : float
        Spectral width of the impurity problem for DLR basis.
    eps : float
        Accuracy of the DLR basis to represent Green’s function

    Returns
    -------
    sigma: imaginary frequency BlockGf
        The self-energy obtained by solving the Dyson equation via CRM.

    References
    ----------
    [1] https://arxiv.org/abs/2310.01266.
    """
    # Fit DLR Green’s function to imaginary time Green’s function
    g_dlr_iw = fit_gf_dlr(g_tau, w_max=w_max, eps=eps)

    # Read off G0 at the DLR nodes
    names = list(g_tau.indices)
    mesh_iw = MeshDLRImFreq(g_dlr_iw.mesh)
    g = Gf(mesh=mesh_iw, target_shape=g_dlr_iw[names[0]].target_shape)
    g0_dlr_iw = BlockGf(name_list=names, block_list=[g, g])
    for name, g in g0_dlr_iw:
        for iwn in mesh_iw:
            g[iwn] = g0_iw[name](iwn.value)

    g = Gf(mesh=mesh_iw, target_shape=g_dlr_iw[names[0]].target_shape)
    sigma_dlr = BlockGf(name_list=names, block_list=[g, g])
    n_iw = g0_iw.mesh.n_iw

    # Use the CRM solver to minimize the Dyson error
    for name, sig in sigma_dlr:
        s_dlr, s_hf, residual = minimize_dyson(
            G0_dlr=g0_dlr_iw[name], G_dlr=g_dlr_iw[name], Sigma_moments=sigma_moments[name]
        )
        sig << s_dlr  # noqa

    # Since a spectral representable G has no constant we have to manually add the Hartree
    # shift after the solver is finished again
    sigma_iw = make_gf_imfreq(sigma_dlr, n_iw=n_iw)
    for name, sig in sigma_iw:
        sig += sigma_moments[name][0]

    return sigma_iw


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

    if solver_params.rebin_g_tau:
        report("Re-binning G(τ)...")
        g_tau_rebinned = solver.G_tau.rebin(solver_params.rebin_n_tau)

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

    if solver_params.correct_hartree:
        # Correct hartree shift
        report("Correcting Hartree shift...")
        densities = dict()
        for spin, g in solver.G_iw:
            # Compute density
            dens = g.density()
            if np.any(dens.imag > 1e-10):
                report("Warning: density is not real")
            densities[spin] = dens.real

        up, dn = params.spin_names
        correction_up = densities[dn] * u
        correction_dn = densities[up] * u
        sigma_iw[up] += correction_up
        sigma_iw[dn] += correction_dn
        report(f"Corrected Hartree shift: {correction_up} (up), {correction_dn} (dn)")

    return sigma_iw, g_l, g_tau_rebinned
