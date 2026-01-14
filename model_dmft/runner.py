# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-14

import os
import re
import selectors
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen, SubprocessError
from typing import Dict, List, TextIO, Tuple, Union

import numpy as np

# noinspection PyPackageRequirements
from h5 import HDFArchive
from triqs.gf import BlockGf, MeshReFreq, Omega, inverse, iOmega_n
from triqs.utility import mpi
from triqs_cpa import (
    G_coherent,
    G_component,
    HilbertTransform,
    initalize_onsite_energy,
    initialize_G_cmpt,
    solve_cpa,
    solve_cpa_fxocc,
)

# from . import cpa
# from .functions import HilbertTransform
from .convergence import calculate_convergences
from .input import InputParameters, get_supported_solvers
from .legendre import check_nl
from .mixer import apply_mixing
from .postprocessing import anacont_pade
from .utility import SIGMA, TIME_FRMT, blockgf, check_broadening, mixing_update, report, symmetrize_gf

EXECUTABLE = sys.executable  # Use current Python executable for subprocesses

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STALL_SECS = 3600  # e.g., 1h

USE_SRUN = os.environ.get("USE_SRUN", "0") == "1"
# MPI_IMPL = os.environ.get("MPI_IMPL", "pmix")  # or "pmi2": check `srun --mpi=list`
# Slurm hints: figure out how many tasks per node were granted
NUM_TASKS = os.environ.get("SLURM_NTASKS") or os.environ.get("SLURM_NTASKS_PER_NODE")

ENV = dict(os.environ)
ENV.setdefault("PYTHONUNBUFFERED", "1")
ENV.setdefault("OMP_NUM_THREADS", "1")
ENV.setdefault("OPENBLAS_NUM_THREADS", "1")
ENV.setdefault("MKL_NUM_THREADS", "1")
ENV.setdefault("VECLIB_MAXIMUM_THREADS", "1")
ENV.setdefault("NUMEXPR_NUM_THREADS", "1")

sel = selectors.DefaultSelector()

Process = Popen[str]


@dataclass
class ProcessInfo:
    out_path: str
    err_path: str
    out: TextIO
    err_lines: List[str]
    cmpt: str


ProcessInfos = Dict[Process, ProcessInfo]


def choose_slurm_mpi_plugin() -> str:
    env = os.environ
    if env.get("MPI_IMPL"):
        return env["MPI_IMPL"]
    # Detect vendor via mpi4py if available
    try:
        from mpi4py import MPI

        name, (major, minor, patch) = MPI.get_vendor()
        if "Open MPI" in name:
            # Pick a conservative default; allow override above
            return "pmix_v5" if major >= 5 else "pmix_v4"
        if "MPICH" in name or "Intel" in name:
            return "pmi2"
    except Exception:
        pass
    # Fallback that works on many MPICH clusters
    return "pmi2"


def report_header(text: str, width: int, char: str = "-") -> None:
    """Print a header with a given width."""
    report(char * width)
    report(f"{' ' + text + ' ':{char}^{width}}")
    report(char * width)


def write_header(file: str, it: int, mode: str) -> None:
    """Write a header with a given width to a file."""
    with open(file, mode) as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f" Iteration {it}\n")
        f.write("=" * 100 + "\n\n")


def print_params(params: InputParameters) -> None:
    """Print the input parameters to the console."""
    report("")
    report("INPUT PARAMETERS")
    report("-" * 60)
    report(f"jobname:        {params.jobname}")
    report(f"location:       {params.location_path}")
    report(f"output:         {params.output_path}")
    report("")
    report(f"lattice:        {params.lattice}")
    report(f"gf struct:      {params.gf_struct}")
    report(f"half bandwidth: {params.half_bandwidth}")
    report(f"conc:           {params.conc}")
    report(f"U:              {params.u}")
    report(f"eps:            {params.eps}")
    report(f"H field:        {params.h_field}")
    report(f"mu:             {params.mu}")
    if params.is_real_mesh:
        report(f"mesh:           {params.w_range}  N: {params.n_w}  eta: {params.eta}")
    else:
        report(f"mesh:           N: {params.n_iw}  beta: {params.beta}")
    report(f"Symmetrize:     {params.symmetrize}")
    report("")
    report(f"CPA tol:        {params.tol_cpa}")
    report(f"G tol:          {params.gtol}")
    report(f"S tol:          {params.stol}")
    report(f"Mixing DMFT:    {params.mixing_dmft}")
    report(f"Mixing CPA:     {params.mixing_cpa}")
    report("")
    if params.solver:
        solver = params.solver_params
        report(f"Solver:         {solver.type}")
        if solver.type == "ftps":
            report(f"Bath fit:       {solver.bath_fit}")
            report(f"Method:         {solver.method}")
            report(f"N bath:         {solver.n_bath}")
            report(f"Time steps:     {solver.time_steps}")
            report(f"dt:             {solver.dt}")
            report(f"Sweeps:         {solver.sweeps}")
            report(f"Trunc weight:   {solver.tw}")
            report(f"Max bond dim:   {solver.maxm}")
            report(f"Max Krylov:     {solver.nmax}")
        elif solver.type == "cthyb":
            report(f"N warmup:       {solver.n_warmup_cycles}")
            report(f"N cycles:       {solver.n_cycles}")
            report(f"Length cycle:   {solver.length_cycle}")
            report(f"Density matrix: {solver.density_matrix}")
            report(f"Tail fit:       {solver.tail_fit}")
            report(f"Fit max moment: {solver.fit_max_moment}")
            report(f"Fit min N:      {solver.fit_min_n}")
            report(f"Fit max N:      {solver.fit_max_n}")
            report(f"Measure G(l):   {solver.measure_g_l}")
            report(f"N_l:            {solver.n_l}")
    report("-" * 60)
    report("")


# Error when the previous iteration data is not compatible with the current input parameters
class IncompatibleDataError(ValueError):
    def __init__(self, message: str):
        super().__init__("Cannot continue with previous data: " + message)


def check_compatible_input(archive_file: Union[Path, str], params: InputParameters) -> None:
    """Check if the input parameters are compatible with the output archive."""
    with HDFArchive(archive_file, "r") as ar:
        old_params = ar["params"]
        if params.lattice != old_params.lattice:
            raise IncompatibleDataError("Lattice mismatch.")
        if params.gf_struct != old_params.gf_struct:
            raise IncompatibleDataError("Green's function structure mismatch.")
        if params.half_bandwidth != old_params.half_bandwidth:
            raise IncompatibleDataError("Half bandwidth mismatch.")
        if params.is_real_mesh:
            if params.w_range != old_params.w_range:
                raise IncompatibleDataError("Frequency range mismatch.")
            if params.n_w != old_params.n_w:
                raise IncompatibleDataError("Number of frequencies mismatch.")
        else:
            if params.n_iw != old_params.n_iw:
                raise IncompatibleDataError("Number of Matsubara frequencies mismatch.")


def load_state(params: InputParameters) -> Tuple[int, BlockGf, BlockGf, BlockGf, float, float]:
    """Load the previous state of the calculation from the output archive.

    Parameters
    ----------
    params : InputParameters
        The input parameters.

    Returns
    -------
    it_prev : int
        The previous iteration number.
    sigma_dmft : BlockGf
        The previous DMFT self-energy.
    sigma_cpa : BlockGf
        The previous CPA self-energy.
    gf_coh : BlockGf
        The previous coherent Green's function.
    occ : float
        The previous occupation numbers.
    """
    location = Path(params.location)
    out_file = str(location / params.output)
    it_prev, sigma_dmft, sigma_cpa, gf_coh, occ, mu = 0, None, None, None, None, None

    if Path(out_file).exists():
        if params.load_iter == 0:
            report(f"Found previous file {out_file}, overwriting file and restarting...")
            # Overwrite previous file
            with HDFArchive(out_file, "w"):
                pass
        else:
            report("Reading data...")
            with HDFArchive(out_file, "a") as ar:
                try:
                    check_compatible_input(out_file, params)
                    it_prev = ar["it"]
                    key_sigma_dmft = "sigma_dmft"
                    key_sigma_cpa = "sigma_cpa"
                    key_gf_coh = "g_coh"
                    key_occ = "occ"
                    key_mu = "mu"
                    if params.load_iter > 0:
                        it = min(it_prev, params.load_iter)
                        if f"sigma_dmft-{it}" in ar:
                            it_prev = it
                            key_sigma_dmft = f"sigma_dmft-{it_prev}"
                            key_sigma_cpa = f"sigma_cpa-{it_prev}"
                            key_gf_coh = f"g_coh-{it_prev}"
                            key_occ = f"occ-{it_prev}"
                        # Remove data of later iterations
                        for key in list(ar.keys()):
                            for i in range(it_prev + 1, ar["it"]):
                                if key.endswith(f"-{i}"):
                                    del ar[key]

                    if key_sigma_dmft in ar:
                        sigma_dmft = ar[key_sigma_dmft]
                    if key_sigma_cpa in ar:
                        sigma_cpa = ar[key_sigma_cpa]
                    if key_gf_coh in ar:
                        gf_coh = ar[key_gf_coh]
                    if key_occ in ar:
                        occ = ar[key_occ]
                    if key_mu in ar:
                        mu = ar[key_mu]
                    if it_prev < params.n_loops:
                        s = f"continuing from previous iteration {it_prev}"
                        report(f"Found previous file {out_file}, {s}...")
                    else:
                        report("Already completed all iterations.")
                except KeyError:
                    pass
    return it_prev, sigma_dmft, sigma_cpa, gf_coh, occ, mu


def update_dataset(archive_file: Union[Path, str], keep_iter: bool = True) -> None:
    """Update the datasets in the output archive.

    If 'keep_iter' is True, the data of the previous iteration is stored in new datasets with the
    iteration number as suffix.

    Parameters
    ----------
    archive_file : Path | str
        The path to the archive file.
    keep_iter : bool, optional
        Whether to keep previous iterations, by default True.
    """
    if not keep_iter:
        return

    with HDFArchive(archive_file, "a") as ar:
        it = ar["it"]
        ar[f"sigma_cpa-{it}"] = ar["sigma_cpa"]
        ar[f"g_coh-{it}"] = ar["g_coh"]
        ar[f"g_cmpt-{it}"] = ar["g_cmpt"]
        ar[f"occ-{it}"] = ar["occ"]
        ar[f"err_g-{it}"] = ar["err_g"]
        ar[f"err_sigma-{it}"] = ar["err_sigma"]
        ar[f"err_occ-{it}"] = ar["err_occ"]
        ar[f"mu-{it}"] = ar["mu"]
        if "sigma_dmft" in ar:
            ar[f"sigma_dmft-{it}"] = ar["sigma_dmft"]
            ar[f"delta-{it}"] = ar["delta"]
        if "sigma_dmft_raw" in ar:
            ar[f"sigma_dmft_raw-{it}"] = ar["sigma_dmft_raw"]

        if "g_coh_real" in ar:
            ar[f"g_coh_real-{it}"] = ar["g_coh_real"]
        if "g_cmpt_real" in ar:
            ar[f"g_cmpt_real-{it}"] = ar["g_cmpt_real"]
        if "sigma_cpa_real" in ar:
            ar[f"sigma_cpa_real-{it}"] = ar["sigma_cpa_real"]
        if "g_ret" in ar:
            ar[f"g_ret-{it}"] = ar["g_ret"]
        if "g_l" in ar:
            ar[f"g_l-{it}"] = ar["g_l"]
        if "g_tau" in ar:
            ar[f"g_tau-{it}"] = ar["g_tau"]


def hybridization(gf: BlockGf, eps: BlockGf, sigma: BlockGf, mu: float, eta: float = 0.0, name: str = "Δ") -> BlockGf:
    """Compute bath hybridization function Δ(z).

    The bath hybridization function is defined as:

    .. math::
        Δ_i(z) = z + μ - ε_i - Σ_i(z) - G_i(z)^{-1}

    where `z` are real frequencies with a complex broadening or Matsubara frequencies.

    Parameters
    ----------
    gf : BlockGf
        The Green's function `G_i(z)`.
    eps : BlockGf
        The effective onsite energies `ε_i`.
    sigma : BlockGf
        The self-energy `Σ_i(z)`.
    mu : float
        The chemical potential.
    eta : float, optional
        The broadening parameter, used for real frequencies. The default is 0.0.
    name : str, optional
        The name of the hybridization function. The default is "Δ".
    """
    x = Omega if isinstance(gf.mesh, MeshReFreq) else iOmega_n
    delta = sigma.copy()
    delta.name = name
    for cmpt, e_i in eps:
        for spin, e_is in e_i:
            delta[cmpt][spin] << x + 1j * eta + mu - e_is - sigma[cmpt][spin] - inverse(gf[cmpt][spin])
    return delta


def solve_impurity(tmp_file: Union[str, Path]) -> None:
    """Solve the impurity problem using the given parameters and data from a temporary file.

    This method is called by the main program in a separate process. It loads the parameters
    and data from a temporary file, solves the impurity problem and writes the results back to
    the temporary file. This is required to run multiple impurity solvers in parallel.

    The solver type is determined by the parameters in the temporary file.

    Available solvers:
    - forkTPS (ftps): Fork tensor product solver
    - cthyb: Continuous-time hybridization expansion solver

    Parameters
    ----------
    tmp_file : str | Path
        The path to the temporary file containing the parameters and data.
    """
    # Load parameters and data from temporary file
    with HDFArchive(str(tmp_file), "r") as ar:
        params = ar["params"]
        u = ar["u"]
        e_onsite = ar["e_onsite"]
        delta = ar["delta"]
        it = ar["it"]

    if u == 0:
        # No interaction, return zero self-energy
        report("Skipping...")
        report("")
        return

    start_time = datetime.now()
    if mpi.is_master_node():
        width = 100
        report("=" * width)
        report(f" Iteration {it}")
        report("=" * width)
        report("")
        report(f"Start: {start_time:{TIME_FRMT}}")

    solver_type = params.solver
    if solver_type == "ftps":
        from .solvers.ftps import solve_ftps

        solver = solve_ftps(params, u, e_onsite, delta)

        # Write results back to temporary file
        mpi.barrier()
        if mpi.is_master_node():
            with HDFArchive(str(tmp_file), "a") as ar:
                ar["solver"] = solver
                ar["sigma_dmft"] = solver.Sigma_w
                ar["g_ret"] = solver.G_ret

    elif solver_type == "cthyb":
        from .solvers.cthyb import postprocess_cthyb, solve_cthyb

        solver = solve_cthyb(params, u, e_onsite, delta)

        mpi.barrier()
        if mpi.is_master_node():
            # Run post-processing of the solver results
            sigma_post, g_l, g_tau_rebinned = postprocess_cthyb(params, solver, u)

            # Write results back to temporary file
            with HDFArchive(str(tmp_file), "a") as ar:
                ar["solver"] = solver
                ar["g_tau_raw"] = solver.G_tau
                if g_tau_rebinned is None:
                    ar["g_tau"] = solver.G_tau
                else:
                    ar["g_tau"] = g_tau_rebinned
                ar["auto_corr_time"] = solver.auto_corr_time
                ar["average_sign"] = solver.average_sign
                ar["average_order"] = solver.average_order
                if sigma_post is not None:
                    ar["sigma_dmft_raw"] = solver.Sigma_iw
                    ar["sigma_dmft"] = sigma_post
                else:
                    ar["sigma_dmft"] = solver.Sigma_iw
                if g_l is not None:
                    # Store the Legendre Green's functions
                    ar["g_l"] = g_l

    elif solver_type == "ctseg":
        from .solvers.ctseg import postprocess_ctseg, solve_ctseg

        solver = solve_ctseg(params, u, e_onsite, delta)

        mpi.barrier()
        if mpi.is_master_node():
            # Run post-processing of the solver results
            g_iw, sigma_iw, g_l = postprocess_ctseg(params, solver, u, e_onsite, delta)

            # Write results back to temporary file
            with HDFArchive(str(tmp_file), "a") as ar:
                ar["solver"] = solver
                ar["g_tau"] = solver.results.G_tau  # type: ignore
                ar["g_iw"] = g_iw
                ar["sigma_dmft"] = sigma_iw
                if g_l is not None:
                    # Store the Legendre Green's functions
                    ar["g_l"] = g_l

    elif solver_type == "hubbardI":
        from .solvers.hubbard1 import solve_hubbard

        solver = solve_hubbard(params, u, e_onsite, delta)

        # Write results back to temporary file
        mpi.barrier()
        if mpi.is_master_node():
            with HDFArchive(str(tmp_file), "a") as ar:
                ar["solver"] = solver
                ar["sigma_dmft"] = solver.Sigma_iw

    elif solver_type == "hartree":
        from .solvers.hartree import solve_hartree

        solver = solve_hartree(params, u, e_onsite, delta)
        sigma = delta.copy()
        for cmpt, delt in delta:
            sigma[cmpt] << solver.Sigma_HF[cmpt].data[0, 0]  # noqa

        # Write results back to temporary file
        mpi.barrier()
        if mpi.is_master_node():
            with HDFArchive(str(tmp_file), "a") as ar:
                ar["solver"] = solver
                ar["sigma_dmft"] = sigma

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    if mpi.is_master_node():
        end_time = datetime.now()
        report("")
        report(f"End:      {end_time:{TIME_FRMT}}")
        report(f"Duration: {end_time - start_time}")
        report("")


def _prepare_tmp_file(
    tmp_file: Union[str, Path],
    params: InputParameters,
    it: int,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
) -> None:
    """Prepare a temporary file for the impurity solver.

    This function writes the parameters and data to a temporary file that can be used by the
    impurity solver in a separate process, e.g. using MPI. This is required to run multiple
    impurity solvers in parallel.
    """
    if not mpi.is_master_node():
        return

    sigma = delta.copy()
    sigma.name = "Σ"
    sigma.zero()
    with HDFArchive(str(tmp_file), "w") as ar:
        ar["params"] = params
        ar["it"] = it
        ar["u"] = u
        ar["e_onsite"] = e_onsite
        ar["delta"] = delta
        ar["sigma_dmft"] = sigma


def _load_tmp_files(sigma_dmft: BlockGf, tmp_filepath: str, archive_file: str) -> None:
    """Load the results from the temporary files of the impurity solvers.

    This function loads the results from the impurity solvers writen to the temporary files.
    The (component) self energies are loaded into the `sigma_dmft` TRIQS Gf object.
    Any additional output data of the solvers are written to the output archive file directly.
    """
    if not mpi.is_master_node():
        return

    g_tau_blocks_raw, g_tau_blocks, g_ret_blocks, g_l_blocks = dict(), dict(), dict(), dict()
    auto_corr_time = dict()
    average_sign = dict()
    average_order = dict()
    sigma_dmft_raw, save_raw = sigma_dmft.copy(), False

    # Load main results from temporary file
    for cmpt, sig in sigma_dmft:
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        # report(f"Loading temporary file {tmp_file}...")
        with HDFArchive(str(tmp_file), "r") as ar:
            if "sigma_dmft" in ar:
                sig << ar["sigma_dmft"]  # noqa
            if "sigma_dmft_raw" in ar:
                save_raw = True
                sigma_dmft_raw[cmpt] << ar["sigma_dmft_raw"]  # noqa
            if "g_ret" in ar:
                g_ret_blocks[cmpt] = ar["g_ret"]
            if "g_l" in ar:
                g_l_blocks[cmpt] = ar["g_l"]
            if "g_tau_raw" in ar:
                g_tau_blocks_raw[cmpt] = ar["g_tau_raw"]
            if "g_tau" in ar:
                g_tau_blocks[cmpt] = ar["g_tau"]
            if "auto_corr_time" in ar:
                auto_corr_time[cmpt] = ar["auto_corr_time"]
            if "average_sign" in ar:
                average_sign[cmpt] = ar["average_sign"]
            if "average_order" in ar:
                average_order[cmpt] = ar["average_order"]

    if save_raw:
        # Write raw sigma_dmft to output file
        with HDFArchive(archive_file, "a") as ar:
            ar["sigma_dmft_raw"] = sigma_dmft_raw

    if g_ret_blocks:
        # Write real time Gf to output file (only for FTPS solver)
        names = list(g_ret_blocks.keys())
        blocks = [g_ret_blocks[name] for name in names]
        gf_ret = blockgf(blocks[0].mesh, names=names, blocks=blocks)
        with HDFArchive(archive_file, "a") as ar:
            ar["g_ret"] = gf_ret

    if g_l_blocks:
        # Write Legendre Gf to output file (only for CTHYB solver)
        names = list(g_l_blocks.keys())
        blocks = [g_l_blocks[name] for name in names]
        g_l = blockgf(blocks[0].mesh, names=names, blocks=blocks)
        with HDFArchive(archive_file, "a") as ar:
            ar["g_l"] = g_l

    if g_tau_blocks_raw:
        # Write G_tau to output file (only for CTHYB solver)
        names = list(g_tau_blocks_raw.keys())
        blocks = [g_tau_blocks_raw[name] for name in names]
        g_tau = blockgf(blocks[0].mesh, names=names, blocks=blocks)
        with HDFArchive(archive_file, "a") as ar:
            ar["g_tau_raw"] = g_tau

    if g_tau_blocks:
        # Write G_tau to output file (only for CTHYB solver)
        names = list(g_tau_blocks.keys())
        blocks = [g_tau_blocks[name] for name in names]
        g_tau = blockgf(blocks[0].mesh, names=names, blocks=blocks)
        with HDFArchive(archive_file, "a") as ar:
            ar["g_tau"] = g_tau

    if auto_corr_time:
        with HDFArchive(archive_file, "a") as ar:
            ar["auto_corr_time"] = auto_corr_time

    if average_sign:
        with HDFArchive(archive_file, "a") as ar:
            ar["average_sign"] = average_sign

    if average_order:
        with HDFArchive(archive_file, "a") as ar:
            ar["average_order"] = average_order

    # Remove temporary files
    for cmpt, sig in sigma_dmft:
        if mpi.is_master_node():
            Path(tmp_filepath.format(cmpt=cmpt)).unlink(missing_ok=True)


def solve_impurities_seq(
    params: InputParameters,
    it: int,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
    sigma_dmft: BlockGf,
) -> None:
    """Solve the impurity problems sequentially.

    This method solves the impurity problems sequentially, i.e. one after the other.
    The method prepares temporary files for each component, solves the impurity problem
    and loads the results back from the temporary files.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    it : int
        The current iteration number. Used for logging.
    u : np.ndarray
        The interaction parameters for each component as a 1D array.
    e_onsite : np.ndarray
        The effective onsite energies for each component as a 1D array.
    delta : BlockGf
        The hybridization function Δ(z) for each component. See the `hybridization` function
        for details.
    sigma_dmft : BlockGf
        The output DMFT self-energy Σ(z) object. The results are written to this TRIQS Gf object.

    See Also
    --------
    solve_impurity : Solve the impurity problem using the parameters and data from a temporary file.
    """
    tmp_dir = Path(params.tmp_dir_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = str(tmp_dir / "tmp-{cmpt}.h5")
    archive_file = str(params.output_path)

    # Write parameters and data to temporary files
    if mpi.is_master_node():
        for i, (cmpt, delt) in enumerate(delta):
            tmp_file = tmp_filepath.format(cmpt=cmpt)
            _prepare_tmp_file(tmp_file, params, it, u[i], e_onsite[i], delta[cmpt])
    mpi.barrier()

    # --- Solve impurity problems -----
    report("Solving impurity problems sequentially...")
    for cmpt in sigma_dmft.indices:
        report(f"Solving component {cmpt}...")
        solve_impurity(tmp_filepath.format(cmpt=cmpt))
    mpi.barrier()
    report("Solvers done!")

    # ---- End solve ------------------

    # Load results back from temporary files and remove them
    _load_tmp_files(sigma_dmft, tmp_filepath, archive_file)


def register_process(infos: ProcessInfos, p: Process, cmpt: str, out_file: str, err_file: str) -> None:
    """Register a process with the selector and store its information."""
    out_fh = open(out_file, "a", buffering=1)
    if p.stdout:
        sel.register(p.stdout, selectors.EVENT_READ, (p, out_fh, "out"))
    if p.stderr:
        sel.register(p.stderr, selectors.EVENT_READ, (p, None, "err"))
    infos[p] = ProcessInfo(out_path=out_file, err_path=err_file, out=out_fh, err_lines=list(), cmpt=cmpt)


def solve_impurities(
    params: InputParameters,
    it: int,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
    sigma_dmft: BlockGf,
    nproc: int,
    verbosity: int = 2,
) -> None:
    """Solve the impurity problems in parallel using MPI.

    This method solves the impurity problems in parallel using MPI. The number of processes is
    determined by the number of components and the total number of processes, hence the number
    of processes per component is `nproc / N_cmpt` where `N_cmpt` is the number of components.
    The method prepares the temporary files for each component, starts the impurity solvers in
    separate processes, waits for all processes to finish and loads the results back from the
    temporary files.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    u : np.ndarray
        The interaction parameters for each component as a 1D array.
    e_onsite : np.ndarray
        The effective onsite energies for each component as a 1D array.
    delta : BlockGf
        The hybridization function Δ(z) for each component. See the `hybridization` function
        for details.
    sigma_dmft : BlockGf
        The output DMFT self-energy Σ(z) object. The results are written to this TRIQS Gf object.
    nproc : int
        The total number of processes to use. The number of processes per component is determined
        by `nproc / N_cmpt` where `N_cmpt` is the number of components. The number of processes
        `nproc` must be divisible by the number of components.
    it : int
        The current iteration number. Used for logging. The default is None.
    verbosity : int, optional
        The verbosity level. The default is 2.

    See Also
    --------
    solve_impurity : Solve the impurity problem using the parameters and data from a temporary file.
    """
    if not mpi.is_master_node():
        return  # This method should only be called from the master node

    solver_type = params.solver
    # Compute number of processes per interacting component
    n_u = sum(1 for ui in u if ui != 0)
    n = nproc / n_u

    # Warn if number of processes is too high for the solver
    if n % 1 != 0:
        raise ValueError("Number of processes must be divisible by number of components.")
    n = max(1, int(n))
    max_proc = params.solver_params.MAX_PROCESSES
    if max_proc is not None and n > max_proc:
        line1 = f"WARNING: Number of processes {n} is too high for solver {solver_type}."
        line2 = f"         Maximum number of processes per solver is {max_proc}."
        line = "-" * max(len(line1), len(line2))
        report("")
        report(line)
        report(line1)
        report(line2)
        report(line)
        report("")

    tmp_dir = Path(params.tmp_dir_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = str(tmp_dir / "tmp-{cmpt}.h5")
    archive_file = str(params.output_path)

    stdout_filepath = str(Path(params.location_path) / "solver-{cmpt}.log")
    stderr_filepath = str(Path(params.location_path) / "solver-err-{cmpt}.log")

    # Write parameters and data to temporary files
    for i, (cmpt, delt) in enumerate(delta):
        u_cmpt = u[i]
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        _prepare_tmp_file(tmp_file, params, it, u_cmpt, e_onsite[i], delta[cmpt])

    # --- Solve impurity problems -----

    procs = list()
    proc_info: ProcessInfos = dict()

    try:
        report("Starting processes...")
        for i, cmpt in enumerate(sigma_dmft.indices):
            u_cmpt = u[i]
            # Prepare tmp/output file paths
            tmp_file = tmp_filepath.format(cmpt=cmpt)
            out_file = stdout_filepath.format(cmpt=cmpt)
            err_file = stderr_filepath.format(cmpt=cmpt)
            # Write header to output file
            # write_header(out_file, it, out_mode)
            # write_header(err_file, it, out_mode)
            if u_cmpt == 0:
                continue
                # base_cmd = list()
            else:
                buff_opt = ["stdbuf", "-oL", "-eL"]
                if USE_SRUN:
                    # ensure TRIQS uses MPI under srun/pmix
                    ENV["TRIQS_FORCE_MPI_INIT"] = "1"
                    # Choose the MPI plugin for srun
                    plugin = choose_slurm_mpi_plugin()
                    prog = ["srun", "-n", str(n)]
                    opts = [
                        "-u",  # unbuffered output
                        "--exact",  # use exact number of tasks, do not oversubscribe
                        f"--mpi={plugin}",  # or pmi2 on older stacks; check `srun --mpi=list`
                    ] + buff_opt
                elif n > 1:
                    prog = ["mpirun", "-np", str(n)]
                    opts = ["--bind-to", "none"] + buff_opt  # do not bind processes to cores
                else:
                    prog = list()
                    opts = list()

            # Start process and register it with the selector
            cmd = prog + opts + [EXECUTABLE, "-u", "-m", "model_dmft", "solve_impurity", tmp_file]
            cmd_str = prog + ["model_dmft", "solve_impurity", tmp_file]
            if verbosity > 0:
                report("> " + " ".join(shlex.quote(x) for x in cmd_str))
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=0, env=ENV)
            register_process(proc_info, p, cmpt, out_file, err_file)
            procs.append(p)

        # Event loop until all concurrent steps finish
        alive = set(procs)
        # Last activity time for each process to avoid stalling
        last_activity = {p: time.monotonic() for p in procs}

        while alive or sel.get_map():
            events = sel.select(timeout=0.1)
            for key, _ in events:
                stream = key.fileobj
                p, fh, kind = key.data
                try:
                    chunk = os.read(stream.fileno(), 1024)  # bytes; non-blocking due to selector
                except BlockingIOError:
                    continue
                # line = stream.readline().rstrip("\r\n")  # type: ignore
                if chunk:
                    # Update activity time for this process
                    last_activity[p] = time.monotonic()

                    text = chunk.decode("utf-8", errors="replace")
                    # Turn carriage-return updates into newlines so they show up in the file
                    text = text.replace("\r", "\n")
                    # Remove ANSI escape codes (e.g. color codes)
                    text = ANSI.sub("", text)
                    if kind == "out":
                        # Write stdout line to file
                        fh.write(text)
                        fh.flush()
                        # Report stdout line to console
                        # if verbosity > 2:
                        #     report(f"[{p.pid}] " + line)
                    else:
                        # collect stderr line in memory
                        proc_info[p].err_lines.append(text)
                else:
                    # EOF on this stream
                    sel.unregister(stream)

            now = time.monotonic()
            for p in list(alive):
                # Check if the process stalled
                if now - last_activity[p] > STALL_SECS:
                    raise SubprocessError(f"Step {proc_info[p].cmpt} stalled > {STALL_SECS}s")

                # Check if still alive
                if p.poll() is not None:
                    # Close stdout file
                    try:
                        proc_info[p].out.flush()
                        proc_info[p].out.close()
                    except Exception:
                        pass
                    # Check exit code and write stderr file only on error
                    rc = p.wait()
                    if rc != 0 and proc_info[p].err_lines:
                        with open(proc_info[p].err_path, "a") as ef:
                            ef.writelines(proc_info[p].err_lines)
                    if rc != 0:
                        raise SubprocessError(f"Step {proc_info[p].cmpt} failed (PID {p.pid}, exit {rc})")
                    alive.remove(p)
                    del proc_info[p]
        report("Processes done!")

    finally:
        # Safety: unregister & close anything left
        for key in list(sel.get_map().values()):
            try:
                sel.unregister(key.fileobj)
            except Exception:
                pass
            try:
                key.fileobj.close()  # type: ignore
            except Exception:
                pass
        for info in list(proc_info.values()):
            fh = info.out
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    # ---- End solve ------------------

    # Load results back from temporary files and remove them
    _load_tmp_files(sigma_dmft, tmp_filepath, archive_file)


def postprocess(params: InputParameters, out_file: str) -> None:
    """Postprocess the results of the DMFT+CPA calculation.

    This method loads the results from the output archive, runs analytic continuation of the Green's
    functions and self-energies to real frequencies, and writes the results back to the output.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    out_file : str
        The path to the output archive file.
    """
    # Postprocessing
    if not params.is_real_mesh:
        if params.pade_params is not None:
            pade_params = params.pade_params
            pade_kwargs = dict(
                w_range=pade_params.w_range,
                n_w=pade_params.n_w,
                n_points=pade_params.n_points,
                eta=pade_params.freq_offset,
            )

            with HDFArchive(out_file, "r") as ar:
                sigma_cpa = ar["sigma_cpa"]
                g_coh = ar["g_coh"]
                g_cmpt = ar["g_cmpt"]

            report("Analytic continuation of G(iw) to G(ω) using Pade approximants...")
            g_coh_real = anacont_pade(g_coh, **pade_kwargs)
            names, blocks = list(), list()
            for cmpt, g in g_cmpt:
                names.append(cmpt)
                blocks.append(anacont_pade(g, **pade_kwargs))
            g_cmpt_real = blockgf(g_coh_real.mesh, names=names, blocks=blocks)

            report("Analytic continuation of Σ(iw) to Σ(ω) using Pade approximants...")
            sigma_cpa_real = anacont_pade(sigma_cpa, **pade_kwargs)

            with HDFArchive(out_file, "a") as ar:
                ar["g_coh_real"] = g_coh_real
                ar["g_cmpt_real"] = g_cmpt_real
                ar["sigma_cpa_real"] = sigma_cpa_real


def solve(params: InputParameters, n_procs: int = 0) -> None:
    """Run the DMFT+CPA self-consistency loop.

    This is the main function that runs the DMFT+CPA self-consistency loop. It initializes the
    DMFT and CPA self-energies, calculates the component Green's functions, solves the impurity
    problems and updates the self-energies. The loop is repeated until convergence or the maximum
    number of iterations is reached.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    n_procs : int, optional
        The number of processes to use for the impurity solvers. If 0, the impurity problems are
        solved sequentially. The number of processes must be divisible by the number of components.
        The default is 0.
    """
    start_time = datetime.now()

    # ---- MODEL PARAMETERS ------------------------------------------------------------------------
    lattice = params.lattice
    gf_struct = params.gf_struct
    conc, u, eps, h_field = params.cast_cmpt()
    target_occ = params.occ
    mu = params.mu
    if mu is None:
        mu = 0.0
    # if params.occ is not None and params.mu is None:
    #     raise NotImplementedError("Chemical potential optimization is not implemented yet.")

    e_onsite = eps + h_field * SIGMA - u / 2  # Shape: ([N_cmpt, N_spin])

    # ---- COMPUTATIONAL PARAMETERS ----------------------------------------------------------------
    location = Path(params.location)
    out_file = params.output_path
    mesh = params.mesh
    n_loops = params.n_loops

    # DMFT parameters
    has_interaction = any(u)
    # mixing_dmft = params.mixing_dmft
    # tol_dmft = params.tol_dmft

    # CPA parameters
    cpa_kwds = {
        "method": params.method_cpa,
        "tol": params.tol_cpa,
        "maxiter": params.maxiter_cpa,
        "verbosity": params.verbosity_cpa,
    }
    if params.mixing_cpa:
        cpa_kwds["mixing"] = params.mixing_cpa

    # Check compatibility of solver
    solver_type = params.solver
    solver_params = params.solver_params
    if solver_type:
        supported = get_supported_solvers()
        if solver_type not in supported:
            raise ValueError(f"Solver {solver_type} is not supported. Use one of {supported}.")

        if params.is_real_mesh:
            if not solver_params.RE_MESH:
                raise ValueError(f"Solver {solver_type} is not compatible with real mesh.")
        else:
            if solver_params.RE_MESH:
                raise ValueError(f"Solver {solver_type} is not compatible with Matsubara mesh.")

        if params.eta <= 0 and solver_type == "ftps":
            params.eta = 6 / solver_params.n_bath
            report(f"Broadening not provided, setting eta={params.eta} for FTPS solver.")
    elif has_interaction:
        raise ValueError("Solver parameters are required if U!=0.")
    eta = params.eta

    # ---- DERIVED PARAMETERS ----------------------------------------------------------------------

    up, dn = params.spin_names
    check_broadening(mesh, eta)
    ht = HilbertTransform(lattice, params.half_bandwidth)

    # Initialize coherent potential (CPA self-energy) witrh zeros
    sigma_cpa = blockgf(mesh, gf_struct=gf_struct, name="Σ_cpa")

    # Initialize DMFT self energies
    sigma_dmft = initialize_G_cmpt(sigma_cpa, cmpts=len(conc))
    for i, (cmpt, sigma) in enumerate(sigma_dmft):
        for spin, sig in sigma:
            sig << u[i] / 2

    # Initialize effective onsite energies as BlockGf
    eps = initalize_onsite_energy(sigma_cpa, conc, e_onsite)
    freq_name = "w" if params.is_real_mesh else "iw"

    # ---- START OF COMPUTATION --------------------------------------------------------------------

    print_params(params)
    report("")
    report(f"Starting job '{params.jobname}' at {start_time:{TIME_FRMT}}")

    it_prev = 0
    sigma_cpa_prev, g_coh_prev, occ_prev = None, None, None
    if mpi.is_master_node():
        location.mkdir(parents=True, exist_ok=True)
        it_prev, sigma_dmft_prev, sigma_cpa_prev, g_coh_prev, occ_prev, mu_prev = load_state(params)
        if it_prev > 0:
            if sigma_dmft_prev:
                for cmpt, sig in sigma_dmft_prev:
                    # noinspection PyStatementEffect
                    sigma_dmft[cmpt] << sig
            if sigma_cpa_prev is not None:
                sigma_cpa << sigma_cpa_prev
            if mu_prev is not None and target_occ is not None:
                mu = mu_prev

    # Broadcast data
    sigma_cpa = mpi.bcast(sigma_cpa)
    sigma_dmft = mpi.bcast(sigma_dmft)
    it_prev = mpi.bcast(it_prev)

    # Save parameters in archive
    if mpi.is_master_node():
        with HDFArchive(out_file, "a") as ar:
            ar["params"] = params

    # Remove any existing solver outputs
    for file in Path(params.location_path).glob("solver-*.log"):
        file.unlink(missing_ok=True)

    # Start iterations
    try:
        for it in range(it_prev + 1, n_loops + 1):
            mpi.barrier()
            iter_start_time = datetime.now()
            report("")
            report_header(f"Iteration {it} / {n_loops}", width=100, char="=")
            report("")
            report(f"Start: {iter_start_time:{TIME_FRMT}}")
            # Calculate local (component) Green's functions
            report(f"Computing component Green's functions G_i({freq_name})...")
            eps_eff = eps + sigma_dmft
            g_cmpt = G_component(ht, sigma_cpa, conc, eps_eff, mu=mu, eta=eta, scale=False)

            # Patch mu in params for solvers
            params.mu = mu

            # Solve impurity problems
            if any(u):
                sigma_old = sigma_dmft.copy()

                # Compute bath hybridization function
                report(f"Computing bath hybdridization function Δ({freq_name})...")
                delta = hybridization(g_cmpt, eps, sigma_dmft, mu, eta=eta)

                # Check mu consistency in delta
                # z - (ϵ - μ) - Δ(z) ≈ Σ(z) + G(z)^(-1)

                if mpi.is_master_node():
                    # z on the mesh
                    any_block = next(iter(g_cmpt))[1]  # g_cmpt is BlockGf over components
                    any_spin_g = next(iter(any_block))[1]  # inner BlockGf over spins
                    mesh = any_spin_g.mesh
                    x = Omega if isinstance(mesh, MeshReFreq) else iOmega_n

                    for cmpt, g_cmpt_block in g_cmpt:
                        for spin, g in g_cmpt_block:
                            # Required Weiss inverse from Dyson: G0^{-1} = Σ + G^{-1}
                            g0inv_req = g.copy()
                            g0inv_req << sigma_dmft[cmpt][spin] + inverse(g)

                            # Weiss inverse implied by your Δ definition: G0^{-1} = z + μ - ε - Δ
                            g0inv_from_delta = g.copy()
                            g0inv_from_delta << (x + mu + 1j * eta) - eps[cmpt][spin] - delta[cmpt][spin]

                            diff = g.copy()
                            diff << g0inv_from_delta - g0inv_req  # noqa

                            max_abs = float(np.max(np.abs(diff.data)))
                            mean_re = float(np.mean(diff.data.real))

                            report(
                                f"Weiss check {cmpt} {spin}: "
                                f"max|ΔG0inv|={max_abs:.6g}, mean Re(ΔG0inv)={mean_re:.6g}"
                            )

                if mpi.is_master_node():
                    with HDFArchive(out_file, "a") as ar:
                        # ar["sigma_dmft"] = sigma_dmft
                        # ar[f"error_dmft"] = err_dmft
                        ar["delta"] = delta
                # Solve impurity problems to obtain new sigma_dmft
                if mpi.is_master_node():
                    report(f"Solving for impurity self-energies Σ_i({freq_name})...")
                    report("")

                num_tries = 0
                while True:
                    num_tries += 1
                    kwargs = dict(params=params, u=u, e_onsite=e_onsite, delta=delta, sigma_dmft=sigma_dmft)
                    if n_procs > 1:
                        solve_impurities(**kwargs, nproc=n_procs, it=it)
                    else:
                        solve_impurities_seq(**kwargs, it=it)

                    if params.solver in ("cthyb", "ctseg"):
                        # Print some statistics from CTQMC solvers
                        with HDFArchive(out_file, "r") as ar:
                            auto_corr_time = ar["auto_corr_time"]
                            average_sign = ar["average_sign"]
                            average_order = ar["average_order"]

                        auto_corr_time_parts = list()
                        average_sign_parts = list()
                        average_order_parts = list()
                        for cmpt, t in auto_corr_time.items():
                            auto_corr_time_parts.append(f"{cmpt}: {t:.4f}")
                        for cmpt, s in average_sign.items():
                            average_sign_parts.append(f"{cmpt}: {s:.4f}")
                        for cmpt, o in average_order.items():
                            average_order_parts.append(f"{cmpt}: {o:.4f}")

                        report("Auto-corr. times: " + ", ".join(auto_corr_time_parts))
                        report("Average sign:     " + ", ".join(average_sign_parts))
                        report("Average order:    " + ", ".join(average_order_parts))
                        if any(abs(s - 1.0) > 0.1 for s in average_sign.values()):
                            report("WARNING: Average sign is low, results may be unreliable!")
                        if any(t > 1.0 for t in auto_corr_time.values()):
                            report("================================================================")
                            report("WARNING: Auto-correlation time is high, simulation may be stuck!")
                            report("         Consider increasing 'length_cycle'!")
                            report("================================================================")

                        if mpi.is_master_node():
                            # If we are in CTHYb solver and use Legendre fit, check nl
                            if (
                                params.solver == "cthyb"
                                and params.solver_params.legendre_fit
                                and params.solver_params.n_l_thresh
                            ):
                                with HDFArchive(out_file, "r") as ar:
                                    g_l = ar["g_l"]

                                n_l_old = params.solver_params.n_l
                                n_l_new = check_nl(g_l, n_l_old, params.solver_params.n_l_thresh)
                                if n_l_new is None or n_l_new < 6 or (n_l_new - n_l_old) <= 2:
                                    # n_l is okay or invalid, nothing to do
                                    break
                                else:
                                    report("----------------------------------------------------------------")
                                    report(f"NOTE: Changing n_l from {n_l_old} to {n_l_new} for better accuracy.")
                                    report("       Restarting solvers...")
                                    report("----------------------------------------------------------------")
                                    params.solver_params.n_l = n_l_new
                            else:
                                break
                    if num_tries >= 3:
                        report("================================================================")
                        report(f"WARNING: Did not find optimal n_l after {num_tries} tries.")
                        report("================================================================")
                        break

                # Symmetrize DMFT self-energies
                if params.symmetrize:
                    for cmpt, sig in sigma_dmft:
                        symmetrize_gf(sig)

                # Apply mixing
                if params.mixing_decay:
                    mixing = mixing_update(it, params.mixing_dmft, params.mixing_min, params.mixing_decay)
                    report(f"Mixing factor for DMFT self-energy: {mixing:.4f}")
                else:
                    mixing = params.mixing_dmft
                for cmpt, sig in sigma_dmft:
                    apply_mixing(sigma_old[cmpt], sig, mixing)

                # Update data of iteration
                if mpi.is_master_node():
                    with HDFArchive(out_file, "a") as ar:
                        ar["sigma_dmft"] = sigma_dmft
                        # ar[f"error_dmft"] = err_dmft
                        # ar["delta"] = delta
            mpi.barrier()
            report("")

            # Solve CPA self-consistent equations and update coherent potential (sigm_cpa)
            # sigma_cpa_old = sigma_cpa.copy()
            eps_eff = eps + sigma_dmft
            if target_occ is not None:
                mu, sigma_cpa_out = solve_cpa_fxocc(
                    ht, sigma_cpa, conc, eps_eff, target_occ, mu0=mu, eta=eta, **cpa_kwds
                )
                sigma_cpa << sigma_cpa_out
            else:
                sigma_cpa << solve_cpa(ht, sigma_cpa, conc, eps_eff, mu=mu, eta=eta, **cpa_kwds)

            # Symmetrize CPA self-energy
            if params.symmetrize:
                symmetrize_gf(sigma_cpa)

            if mpi.is_master_node():
                report(f"Computing coherent Green's function G_c({freq_name})...")
                g_coh = G_coherent(ht, sigma_cpa, mu=mu, eta=eta)

                report(f"Computing component Green's functions G_i({freq_name})...")
                g_cmpt = G_component(ht, sigma_cpa, conc, eps_eff, mu=mu, eta=eta, scale=False)

                # Compute occupation numbers
                density = g_coh.density()
                occ_up = density[up][0, 0].real
                occ_dn = density[dn][0, 0].real
                occ = g_coh.total_density().real
                # Check convergence and copy data for next iteration
                err_g, err_sigma, err_occ = calculate_convergences(
                    g_coh, sigma_cpa, occ, g_coh_prev, sigma_cpa_prev, occ_prev, relative=True
                )
                sigma_cpa_prev, g_coh_prev, occ_prev = sigma_cpa.copy(), g_coh.copy(), occ

                # report("")
                # report(f"Writing results of iteration {it} to {out_file}")
                with HDFArchive(out_file, "a") as ar:
                    ar["it"] = it
                    ar["sigma_cpa"] = sigma_cpa
                    ar["g_coh"] = g_coh
                    ar["g_cmpt"] = g_cmpt
                    ar["occ"] = occ
                    ar["err_g"] = err_g
                    ar["err_sigma"] = err_sigma
                    ar["err_occ"] = err_occ
                    ar["mu"] = mu

                # Run postprocessing
                postprocess(params, out_file)

                # Set the latest datasets and remove previous iterations if not needed
                update_dataset(out_file, keep_iter=True)

                report("")
                report(f"Occupation:   total={occ:.4f} up={occ_up:.4f} dn={occ_dn:.4f} (mu={mu:.4f})")
                s = "Iteration: {it:>2} Error-G: {g:.10f} Error-Σ: {sig:.10f} Error-n: {occ:.10f}"
                report(s.format(it=it, g=err_g, sig=err_sigma, occ=err_occ))

                iter_end_time = datetime.now()
                report(f"End:       {iter_end_time:{TIME_FRMT}}")
                report(f"Duration:  {iter_end_time - iter_start_time}")
                if not has_interaction:
                    report("")
                    report("No interaction, skipping further iterations.")
                    report("")
                    break

                # Check if last n iterations converged
                if it > params.n_conv:
                    converged = False
                    # Load n previous errors from archive
                    errors = dict(sigma=list(), gf=list(), occ=list())
                    conv_names = ["Σ", "G", "Occupation"]
                    tolerances = [params.stol, params.gtol, params.occ_tol]
                    with HDFArchive(out_file, "r") as ar:
                        for it_hist in range(it - params.n_conv, it + 1):
                            errors["sigma"].append(ar[f"err_sigma-{it_hist}"])
                            errors["gf"].append(ar[f"err_g-{it_hist}"])
                            errors["occ"].append(ar[f"err_occ-{it_hist}"])

                    # Check if all errors are below tolerance and are decreasing
                    for key, name, tol in zip(errors.keys(), conv_names, tolerances):
                        errs = errors[key]
                        if tol and all(err < tol for err in errs):
                            # Make sure error is decreasing
                            if all(errs[i + 1] < errs[i] for i in range(params.n_conv - 1)):
                                now = datetime.now()
                                report("")
                                report(f"{name} converged in {it} iterations at {now:{TIME_FRMT}}")
                                report("")
                                converged = True
                                break

                    if converged:
                        # Stop iterations if converged
                        break

                # if params.stol and err_sigma < params.stol:
                #     now = datetime.now()
                #     report("")
                #     report(f"Σ converged in {it} iterations at {now:{TIME_FRMT}}")
                #     report("")
                #     break
                # if params.gtol and err_g < params.gtol:
                #     now = datetime.now()
                #     report("")
                #     report(f"G converged in {it} iterations at {now:{TIME_FRMT}}")
                #     report("")
                #     break
                # if params.occ_tol and err_occ < params.occ_tol:
                #     now = datetime.now()
                #     report("")
                #     report(f"Occupation converged in {it} iterations at {now:{TIME_FRMT}}")
                #     report("")
                #     break

        # Postprocessing
        postprocess(params, out_file)

        # Write output files as plain text
        # report("Writing output files...")
        # write_out_files(params)

        end_time = datetime.now()
        run_duration = end_time - start_time
        report("")
        report(f"Finished job '{params.jobname}' at {end_time:{TIME_FRMT}}")
        report(f"Total time: {run_duration}")

    finally:
        try:
            tmp_path = Path(params.tmp_dir_path)
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
        except Exception as e:
            report(f"Could not remove tmp-dir: {e}")
