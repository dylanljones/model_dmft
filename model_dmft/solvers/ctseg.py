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
    make_hermitian,
)
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi

from ..input import CtSegSolverParams, InputParameters
from ..legendre import apply_legendre_filter
from ..utility import blockgf, report


def solve_ctseg(params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf) -> triqs_ctseg.Solver:
    up, dn = params.spin_names
    solver_params: CtSegSolverParams = params.solver_params
    # gf_struct = params.gf_struct
    mu = params.mu

    report("Initializing CTSEG solver...")

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
        "measure_F_tau": solver_params.measure_f_tau,
        "measure_densities": True,
    }

    seed_base = solver_params.random_seed if solver_params.random_seed is not None else 34788
    solve_kwargs["random_seed"] = seed_base + 928374 * mpi.rank  # random seed on each core
    if solver_params.random_name:
        solve_kwargs["random_name"] = solver_params.random_name

    # Initialize solver
    solver = triqs_ctseg.Solver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        # n_iw=params.n_iw,
        n_tau=solver_params.n_tau,
        # delta_interface=True,
        # n_l=solver_params.n_l,
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


def postprocess_ctseg(
    params: InputParameters,
    solver: triqs_ctseg.Solver,
    u: float,
    e_onsite: np.ndarray,
    delta: BlockGf,
) -> Tuple[BlockGf, BlockGf, Union[BlockGf, None]]:
    up, dn = params.spin_names
    solver_params = params.solver_params
    gf_struct = params.gf_struct
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)

    # Initialize delta interface
    g0_iw = delta.copy()
    h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n - delta[name] - h_loc0_mat[i])

    g_tau = solver.results.G_tau
    f_tau = solver.results.F_tau

    iw = MeshImFreq(beta=params.beta, S="Fermion", n_max=params.n_iw)
    g_iw = blockgf(iw, gf_struct=gf_struct, name="G_iw")
    f_iw = blockgf(iw, gf_struct=gf_struct, name="F_iw")

    g_iw << Fourier(g_tau)
    f_iw << Fourier(f_tau)

    sigma_iw = inverse(g_iw) * f_iw

    g_l = None
    if solver_params.legendre_fit:
        # Compute Legendre Gf by filtering binned imaginary time Green's function
        report("Applying Legendre filter...")
        g_l = apply_legendre_filter(g_tau, order=solver_params.n_l)

        # Fit the Green's functions and self energy using the Legendre Green's function
        report("Performing Legendre fit...")
        g_iw_l, g_tau_l, sigma_iw_l = legendre_fit(g0_iw, g_iw, g_tau, g_l)

        sigma_iw = sigma_iw_l

    # if solver_params.correct_hartree:
    #     # Correct hartree shift
    #     report("Correcting Hartree shift...")
    #     densities = dict()
    #     for spin, g in g_iw:
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

    return g_iw, sigma_iw, g_l
