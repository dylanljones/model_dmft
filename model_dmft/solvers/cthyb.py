# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-06

import numpy as np
import triqs.operators as ops
import triqs_cthyb
from triqs.gf import BlockGf, Fourier
from triqs.utility import mpi

from ..input import CthybSolverParams, InputParameters
from ..utility import report


def solve_cthyb(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> BlockGf:
    up, dn = params.spin_names
    solver_params: CthybSolverParams = params.solver_params

    report("Initializing solver.")

    # Local Hamiltonian and interaction term
    h_loc0 = e_onsite[0] * ops.n(up, 0) + e_onsite[1] * ops.n(dn, 0)
    h_int = u * ops.n(up, 0) * ops.n(dn, 0)

    solve_kwargs = {
        "n_warmup_cycles": solver_params.n_warmup_cycles,
        "n_cycles": solver_params.n_cycles,
        "length_cycle": solver_params.length_cycle,
        "perform_tail_fit": solver_params.tail_fit,
        "measure_G_l": solver_params.measure_g_l,
    }
    if solver_params.density_matrix:
        solve_kwargs["measure_density_matrix"] = True
        solve_kwargs["use_norm_as_weight"] = True

    if solver_params.tail_fit:
        if solver_params.fit_min_n == 0:
            solve_kwargs["fit_min_n"] = int(0.5 * params.n_iw)
        else:
            solve_kwargs["fit_min_n"] = solver_params.fit_min_n
        if solver_params.fit_max_n == 0:
            solve_kwargs["fit_max_n"] = params.n_iw
        else:
            solve_kwargs["fit_max_n"] = solver_params.fit_max_n
        solve_kwargs["fit_max_moment"] = solver_params.fit_max_moment
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
        delta_interface=True,
        n_l=solver_params.n_l,
    )
    # Set hybridization function (imaginary time)
    solver.Delta_tau << Fourier(delta)  # type: ignore
    mpi.barrier()

    # Solve impurity problem
    solver.solve(h_loc0=h_loc0, h_int=h_int, **solve_kwargs)
    report("Done!")
    report("")

    return solver.Sigma_iw
