# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-06


import numpy as np
import triqs.operators as ops
import triqs_ctseg
from triqs.gf import BlockGf, Fourier, fit_hermitian_tail, inverse, iOmega_n
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi

from ..input import CtSegSolverParams, InputParameters
from ..utility import report


def solve_ctseg(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> triqs_ctseg.Solver:
    up, dn = params.spin_names
    solver_params: CtSegSolverParams = params.solver_params
    gf_struct = params.gf_struct

    report("Initializing CTSEG solver...")

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
        "measure_F_tau": solver_params.measure_f_tau,
    }
    if solver_params.density_matrix:
        solve_kwargs["measure_densities"] = True
        # solve_kwargs["use_norm_as_weight"] = True

    if solver_params.tail_fit or solver_params.crm_dyson:
        # Used for calculating moments
        solve_kwargs["measure_densities"] = True
        # solve_kwargs["use_norm_as_weight"] = True

    # Different random seed on each core
    solve_kwargs["random_seed"] = 34788 + 928374 * mpi.rank  # Default random seed

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
    for spin, delt in delta:
        tail, err = fit_hermitian_tail(delt)
        delt << delt - tail[0]  # noqa

    solver.Delta_tau << Fourier(delta)  # type: ignore
    # solver.G0_iw << g0_iw  # noqa
    mpi.barrier()

    # Solve impurity problem
    report("Solving impurity...")
    solver.solve(h_loc0=h_loc0, h_int=h_int, **solve_kwargs)
    report("Done!")
    report("")

    return solver
