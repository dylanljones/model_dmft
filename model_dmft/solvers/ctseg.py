# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-06

from typing import Tuple, Union

import numpy as np
import triqs.operators as ops
import triqs_ctseg
from triqs.gf import (
    BlockGf,
    Fourier,
    MeshImFreq,
    inverse,
    iOmega_n,
)
from triqs.gf.gf_fnt import fit_hermitian_tail

# from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi

from ..input import CtSegSolverParams, InputParameters
from ..stabilize import apply_legendre_filter, legendre_fit, pick_nl_opt, truncate_g_l
from ..utility import blockgf, rebin_gf_tau, report


def solve_ctseg(params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta_iw: BlockGf) -> triqs_ctseg.Solver:
    up, dn = params.spin_names
    solver_params: CtSegSolverParams = params.solver_params
    # gf_struct = params.gf_struct
    mu = params.mu
    mu_imp = mu + u / 2

    report("Initializing CTSEG solver...")

    # Local Hamiltonian and interaction term
    eps = e_onsite - mu_imp
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)
    h_loc0 = eps[0] * ops.n(up, 0) + eps[1] * ops.n(dn, 0)

    # h_loc0 = eps[0] * ops.n(up, 0) + eps[1] * ops.n(dn, 0)
    # h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)

    solve_kwargs = {
        "n_warmup_cycles": solver_params.n_warmup_cycles,
        "n_cycles": solver_params.n_cycles,
        "length_cycle": solver_params.length_cycle,
        "measure_F_tau": True,
        "measure_densities": True,
    }

    seed_base = solver_params.random_seed if solver_params.random_seed is not None else 34788
    solve_kwargs["random_seed"] = seed_base + 928374 * mpi.rank  # random seed on each core
    if solver_params.random_name:
        solve_kwargs["random_name"] = solver_params.random_name  # type: ignore

    # Initialize solver
    solver = triqs_ctseg.Solver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        n_tau=solver_params.n_tau,
    )

    # Set hybridization function (imaginary time)
    for spin, delt in delta_iw:
        delt << delt + u / 2  # noqa
        tail, err = fit_hermitian_tail(delt)
        delt << delt - tail[0]  # noqa
    solver.Delta_tau << Fourier(delta_iw)  # type: ignore

    # solver.G0_iw << g0_iw  # noqa
    mpi.barrier()

    # Solve impurity problem
    report("Solving impurity...")
    solver.solve(h_loc0=h_loc0, h_int=h_int, **solve_kwargs)
    report("Done!")
    report("")

    return solver


def postprocess_ctseg(
    params: InputParameters,
    solver: triqs_ctseg.Solver,
    u: float,
    e_onsite: np.ndarray,
    delta_iw: BlockGf,
) -> Tuple[BlockGf, BlockGf, BlockGf, Union[BlockGf, None], Union[BlockGf, None], Union[BlockGf, None]]:
    solver_params: CtSegSolverParams = params.solver_params
    gf_struct = params.gf_struct
    mu = params.mu

    # Allocate DMFT objects
    iw = MeshImFreq(beta=params.beta, S="Fermion", n_max=params.n_iw)
    g0_iw = blockgf(mesh=iw, name="G0_iw", gf_struct=gf_struct)
    g_iw = blockgf(mesh=iw, name="G_iw", gf_struct=gf_struct)
    f_iw = blockgf(mesh=iw, name="F_iw", gf_struct=gf_struct)
    sigma_iw = blockgf(mesh=iw, name="Sigma_iw", gf_struct=gf_struct)

    g_tau = solver.results.G_tau
    f_tau = solver.results.F_tau

    # Set hybridization function (imaginary time)
    for spin, delt in delta_iw:
        delt << delt + u / 2  # noqa
        tail, err = fit_hermitian_tail(delt)
        delt << delt - tail[0]  # noqa

    # Weiss field for the impurity
    # G0^{-1} = iω + μ - ϵ_imp - Δ
    for i, (name, g0) in enumerate(g0_iw):
        g0 << inverse(iOmega_n + mu - e_onsite[i] - delta_iw[name])

    # Fourier transform G_tau and F_tau to get Sigma
    g_iw << Fourier(g_tau)
    f_iw << Fourier(f_tau)
    sigma_iw << inverse(g_iw) * f_iw - u / 2.0
    sigma_iw_raw = sigma_iw.copy()

    #
    # Sigma stabilizations
    #

    g_l = None
    g_tau_rebinned = None
    if solver_params.rebin_tau:
        report("Re-binning G(τ)...")
        g_tau_rebinned = rebin_gf_tau(g_tau, solver_params.rebin_tau)

    if solver_params.legendre_fit:
        # Compute Legendre Gf by filtering binned imaginary time Green's function
        report("Applying Legendre filter...")
        order = solver_params.n_l or solver_params.nl_max or 100
        g_l = apply_legendre_filter(g_tau, order=order)

        if solver_params.n_l is None:
            report("\nOptimizing n_l for Legendre fit...\n\n")
            res = pick_nl_opt(
                g_l,
                g0_iw,
                g_iw,
                sigma_moments=None,
                nl_step=params.solver_params.nl_step,
                smooth=8,
                smooth_err=30,
                iw_noise=solver_params.legendre_iw_noise,
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
                g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(g0_iw, g_iw, g_tau, g_l_trunc)
            else:
                n_l = res.nl
                g_l = truncate_g_l(g_l, n_l)
                report(f"\nOptimal w_max found: {n_l}")
                sigma_iw_l = res.sigma_opt
        else:
            # Fit the Green's functions and self energy using the Legendre Green's function
            report("Performing Legendre fit...")
            g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(g0_iw, g_iw, g_tau, g_l)
        sigma_iw = sigma_iw_l
    else:
        sigma_iw = sigma_iw_raw
        sigma_iw_raw = None

    return g0_iw, g_iw, sigma_iw, sigma_iw_raw, g_tau_rebinned, g_l
