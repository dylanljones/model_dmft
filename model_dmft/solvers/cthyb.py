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
from triqs.gf.tools import fit_legendre as _fit_legendre
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi
from triqs_cthyb.tail_fit import tail_fit

from ..input import CthybSolverParams, InputParameters
from ..utility import report


def solve_cthyb(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> triqs_cthyb.Solver:
    up, dn = params.spin_names
    solver_params: CthybSolverParams = params.solver_params
    gf_struct = params.gf_struct

    report("Initializing CTHYB solver...")

    # Local Hamiltonian and interaction term
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)

    # Initialize delta interface
    g0_iw = delta.copy()
    h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n - delta[name] - h_loc0_mat[i])

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

    # Different random seed on each core
    solve_kwargs["random_seed"] = 34788 + 928374 * mpi.rank  # Default random seed

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


def apply_legendre_filter(g_tau: BlockGf, order: int = 100, g_l_cut: float = 1e-19) -> BlockGf:
    """Filter binned imaginary time Green's function using a Legendre filter.

    Parameters
    ----------
    g_tau : TRIQS imaginary time Block Green's function
    order : int
        Legendre expansion order in the filter
    g_l_cut : float
        Legendre coefficient cut-off

    Returns
    -------
    g_l : TRIQS Legendre Block Green's function
        Fitted Green's function on a Legendre mesh
    """
    l_g_l = []
    for _, g in g_tau:
        g_l = _fit_legendre(g, order=order)
        g_l.data[:] *= np.abs(g_l.data) > g_l_cut
        g_l.enforce_discontinuity(np.identity(g.target_shape[0]))
        l_g_l.append(g_l)
    g_l = BlockGf(name_list=list(g_tau.indices), block_list=l_g_l, name="G_l")
    return g_l


def legendre_fit(
    g0_iw: BlockGf, g_iw: BlockGf, g_tau: BlockGf, g_l: BlockGf
) -> Tuple[BlockGf, BlockGf, BlockGf]:
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
    params: InputParameters, solver: triqs_cthyb.Solver
) -> Tuple[Union[BlockGf, None], Union[BlockGf, None]]:
    """Postprocess the CTHYB solver output.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    solver : triqs_cthyb.Solver
        The CTHYB solver object.

    Returns
    -------
    sigma_iw : BlockGf
        The self-energy obtained by postprocessing the CTHYB solver output.
    g_l : BlockGf
        The Legendre Green's function used in the postprocessing. Only returned if Legendre fitting
         is used.
    """
    solver_params = params.solver_params
    # Output quantities
    sigma_iw = None
    g_l = None

    if solver_params.legendre_fit:
        if solver_params.measure_g_l:
            # G_l measured, use to fit tail of Sigma
            g_l = solver.G_l
        else:
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

        sigma_iw_crm = crm_solve_dyson(
            solver.G_tau, solver.G0_iw, sigma_moments, solver_params.crm_wmax, solver_params.crm_eps
        )

        sigma_iw = sigma_iw_crm

    return sigma_iw, g_l
