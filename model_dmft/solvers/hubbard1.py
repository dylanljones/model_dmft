# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-04

import numpy as np
import triqs.operators as ops
import triqs_hubbardI
from triqs.gf import BlockGf, inverse, iOmega_n
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi

from ..input import HubbardISolverParams, InputParameters
from ..utility import report


def solve_hubbard(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf, sigma: BlockGf
) -> triqs_hubbardI.Solver:
    up, dn = params.spin_names
    solver_params: HubbardISolverParams = params.solver_params
    gf_struct = params.gf_struct

    # Local Hamiltonian and interaction term
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)

    report("Initializing HubbardI solver...")

    # Initialize delta interface
    g0_iw = delta.copy()
    h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n - delta[name] - h_loc0_mat[i])

    solve_kwargs = {
        "calc_gtau": solver_params.measure_g_tau,
        "calc_gl": solver_params.measure_g_l,
        "calc_dm": solver_params.density_matrix,
    }

    solver = triqs_hubbardI.Solver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        n_iw=params.n_iw,
        n_tau=solver_params.n_tau,
        n_l=solver_params.n_l,
    )
    # Fill the Weiss field
    solver.G0_iw << g0_iw  # noqa
    mpi.barrier()

    # Solve impurity problem
    report("Solving impurity...")
    solver.solve(h_int=h_int, **solve_kwargs)

    report("Done!")
    report("")

    return solver
