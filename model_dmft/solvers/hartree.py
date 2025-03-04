# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-04

import numpy as np
import triqs.operators as ops
import triqs_hartree_fock
from triqs.gf import BlockGf, inverse, iOmega_n
from triqs.operators.util.extractors import block_matrix_from_op
from triqs.utility import mpi

from ..input import HartreeSolverParams, InputParameters
from ..utility import report


def solve_hartree(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> triqs_hartree_fock.ImpuritySolver:
    up, dn = params.spin_names
    solver_params: HartreeSolverParams = params.solver_params
    gf_struct = params.gf_struct

    # Local Hamiltonian and interaction term
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)

    report("Initializing Hartree-Fock solver...")

    # Initialize delta interface
    g0_iw = delta.copy()
    h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)
    for i, name in enumerate(delta.indices):
        g0_iw[name] << inverse(iOmega_n - delta[name] - h_loc0_mat[i])

    solve_kwargs = {
        "with_fock": solver_params.with_fock,
        "one_shot": solver_params.one_shot,
        "method": solver_params.method,
        "tol": solver_params.tol,
    }

    solver = triqs_hartree_fock.ImpuritySolver(
        beta=params.beta,
        gf_struct=params.gf_struct,
        n_iw=params.n_iw,
        force_real=solver_params.force_real,
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
