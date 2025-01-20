# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-14

from pathlib import Path
from typing import Union

import numpy as np
import triqs.operators as ops
from forktps import Solver
from forktps.DiscreteBath import SigmaDyson
from forktps.solver import DMRGParams, TevoParams
from forktps.solver_core import Bath, HInt, Hloc  # noqa
from h5 import HDFArchive
from triqs.gf import BlockGf, MeshReFreq, Omega, inverse, iOmega_n
from triqs.utility import mpi

from .ftps import check_bath, construct_bath
from .input import InputParameters
from .utility import DN, UP, report


def hybridization(
    gf: BlockGf, eps: BlockGf, sigma: BlockGf, eta: float = 0.0, name: str = "Δ"
) -> BlockGf:
    """Compute bath hybridization function."""
    x = Omega if isinstance(gf.mesh, MeshReFreq) else iOmega_n
    delta = sigma.copy()
    delta.name = name
    for cmpt, e_i in eps:
        for spin, e_is in e_i:
            delta[cmpt][spin] << x + 1j * eta - e_is - sigma[cmpt][spin] - inverse(gf[cmpt][spin])
    return delta


def prepare_tmp_file(
    tmp_file: Union[str, Path],
    params: InputParameters,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
) -> None:
    sigma = delta.copy()
    sigma.name = "Σ"
    sigma.zero()

    with HDFArchive(str(tmp_file), "w") as ar:
        ar["params"] = params
        ar["u"] = u
        ar["e_onsite"] = e_onsite
        ar["delta"] = delta
        ar["sigma"] = sigma


def _solve_ftps(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> BlockGf:
    gf_struct = params.gf_struct
    mesh = delta.mesh
    solve_params = params.solver_params
    # Prepare parameters for solver
    bath_fit = solve_params["bath_fit"]
    nbath = solve_params["n_bath"]
    common = dict(tw=solve_params["tw"], maxm=solve_params["maxm"], nmax=solve_params["nmax"])
    tevo = TevoParams(
        time_steps=solve_params["time_steps"],
        dt=solve_params["dt"],
        method=solve_params["method"],
        **common,
    )
    dmrg = DMRGParams(sweeps=solve_params["sweeps"], **common)
    solve_kwds = {
        "eta": params.eta,
        "tevo": tevo,
        "params_GS": dmrg,
        "params_partSector": dmrg,
        "measurements": [ops.n(UP, 0), ops.n(DN, 0)],  # Only measure impurity Gfs
    }
    tmp_dir = params.tmp_dir_path
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    if not tmp_dir.endswith("/"):
        tmp_dir += "/"
    solve_kwds["state_storage"] = tmp_dir

    # Construct bath
    report("Constructing bath.")
    bath = construct_bath(delta, params.eta, nbath, bath_fit=bath_fit)
    if not check_bath(bath, delta, params.eta, plot=False):
        raise ValueError("Bath construction failed!")

    # Construct local and interaction Hamiltonian
    report("Initializing solver.")
    hloc = Hloc(gf_struct)
    hloc.Fill(UP, [[e_onsite[0]]])
    hloc.Fill(DN, [[e_onsite[1]]])
    hint = HInt(u=u, j=0.0, up=0.0, dd=True)

    # Initialize solver
    solver = Solver(gf_struct, mesh.omega_min, mesh.omega_max, len(mesh))
    solver.b = bath  # Add bath to solver
    solver.e0 = hloc  # Add local Hamiltonian to solver

    # Run solver
    report("Solving impurity...")
    solver.solve(h_int=hint, **solve_kwds)

    # Update self energy using Dyson equation
    mpi.barrier()
    solver.Sigma_w << SigmaDyson(
        Gret=solver.G_ret,
        bath=solver.b,
        hloc=solver.e0,
        mesh=solver.G_w.mesh,
        eta=params.eta,
    )
    report("Done!")
    report("")

    return solver.Sigma_w


def solve_impurity(tmp_file: Union[str, Path]) -> None:
    # Load parameters and data from temporary file
    with HDFArchive(str(tmp_file), "r") as ar:
        params = ar["params"]
        u = ar["u"]
        e_onsite = ar["e_onsite"]
        delta = ar["delta"]

    if u == 0:
        # No interaction, return zero self-energy
        report("Skipping...", fg="lk")
        report("")
        return

    solver_type = params.solver
    if solver_type == "ftps":
        sigma_dmft = _solve_ftps(params, u, e_onsite, delta)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Write results back to temporary file
    mpi.barrier()
    if mpi.is_master_node():
        with HDFArchive(str(tmp_file), "a") as ar:
            ar["sigma_dmft"] = sigma_dmft
