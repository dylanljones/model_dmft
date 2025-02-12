# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-14

import sys
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen, SubprocessError
from typing import Tuple, Union

import numpy as np

# noinspection PyPackageRequirements
from h5 import HDFArchive
from triqs.gf import BlockGf, MeshReFreq, Omega, inverse, iOmega_n
from triqs.utility import mpi

from . import cpa
from .functions import HilbertTransform
from .input import InputParameters, get_supported_solvers
from .output import write_out_files
from .utility import (
    SIGMA,
    apply_mixing,
    blockgf,
    check_broadening,
    check_convergence,
    report,
    symmetrize_gf,
)

USE_SRUN = Path.home().resolve().parts[1].lower() == "hpc"


def report_header(text: str, width: int, char: str = "-", fg: str = "") -> None:
    """Print a header with a given width."""
    report(char * width, fg=fg)
    report(f"{' ' + text + ' ':{char}^{width}}", fg=fg)
    report(char * width, fg=fg)


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


def check_compatible_input(archive_file: Union[Path, str], params: InputParameters) -> None:
    """Check if the input parameters are compatible with the output archive."""
    with HDFArchive(archive_file, "r") as ar:
        old_params = ar["params"]
        if params.lattice != old_params.lattice:
            raise ValueError("Lattice mismatch.")
        if params.gf_struct != old_params.gf_struct:
            raise ValueError("Green's function structure mismatch.")
        if params.half_bandwidth != old_params.half_bandwidth:
            raise ValueError("Half bandwidth mismatch.")
        if params.is_real_mesh:
            if params.w_range != old_params.w_range:
                raise ValueError("Frequency range mismatch.")
            if params.n_w != old_params.n_w:
                raise ValueError("Number of frequencies mismatch.")
        else:
            if params.n_iw != old_params.n_iw:
                raise ValueError("Number of Matsubara frequencies mismatch.")


def load_state(params: InputParameters) -> Tuple[int, BlockGf, BlockGf]:
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
    """
    location = Path(params.location)
    out_file = str(location / params.output)
    it_prev, sigma_dmft, sigma_cpa = 0, None, None

    if Path(out_file).exists():
        if params.restart:
            report(f"Found previous file {out_file}, overwriting file and restarting...")
            # Overwrite previous file
            with HDFArchive(out_file, "w"):
                pass
        else:
            report("Reading data...")
            with HDFArchive(out_file, "r") as ar:
                try:
                    check_compatible_input(out_file, params)
                    it_prev = ar["it"]
                    key_sigma_dmft = "sigma_dmft"
                    key_sigma_cpa = "sigma_cpa"
                    if params.load_iter > 0:
                        it = min(it_prev, params.load_iter)
                        if f"sigma_dmft-{it}" in ar:
                            it_prev = it
                            key_sigma_dmft = f"sigma_dmft-{it_prev}"
                            key_sigma_cpa = f"sigma_cpa-{it_prev}"
                    if key_sigma_dmft in ar:
                        sigma_dmft = ar[key_sigma_dmft]
                    if key_sigma_cpa in ar:
                        sigma_cpa = ar[key_sigma_cpa]
                    if it_prev < params.n_loops:
                        s = f"continuing from previous iteration {it_prev}"
                        report(f"Found previous file {out_file}, {s}...")
                    else:
                        report("Already completed all iterations.")
                except KeyError:
                    pass
    return it_prev, sigma_dmft, sigma_cpa


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
        if "sigma_dmft" in ar:
            ar[f"sigma_dmft-{it}"] = ar["sigma_dmft"]
            ar[f"delta-{it}"] = ar["delta"]

        if "g_coh_real" in ar:
            ar[f"g_coh_real-{it}"] = ar["g_coh_real"]
        if "g_cmpt_real" in ar:
            ar[f"g_cmpt_real-{it}"] = ar["g_cmpt_real"]
        if "sigma_cpa_real" in ar:
            ar[f"sigma_cpa_real-{it}"] = ar["sigma_cpa_real"]


def hybridization(
    gf: BlockGf, eps: BlockGf, sigma: BlockGf, eta: float = 0.0, name: str = "Δ"
) -> BlockGf:
    """Compute bath hybridization function Δ(z).

    The bath hybridization function is defined as:

    .. math::
        Δ_i(z) = z - ε_i - Σ_i(z) - G_i(z)^{-1}

    where `z` are real frequencies with a complex broadening or Matsubara frequencies.

    Parameters
    ----------
    gf : BlockGf
        The Green's function `G_i(z)`.
    eps : BlockGf
        The effective onsite energies `ε_i`.
    sigma : BlockGf
        The self-energy `Σ_i(z)`.
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
            delta[cmpt][spin] << x + 1j * eta - e_is - sigma[cmpt][spin] - inverse(gf[cmpt][spin])
    return delta


def prepare_tmp_file(
    tmp_file: Union[str, Path],
    params: InputParameters,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
) -> None:
    """Prepare a temporary file for the impurity solver.

    This function writes the parameters and data to a temporary file that can be used by the
    impurity solver in a separate process, e.g. using MPI. This is required to run multiple
    impurity solvers in parallel.
    """
    sigma = delta.copy()
    sigma.name = "Σ"
    sigma.zero()
    with HDFArchive(str(tmp_file), "w") as ar:
        ar["params"] = params
        ar["u"] = u
        ar["e_onsite"] = e_onsite
        ar["delta"] = delta
        ar["sigma"] = sigma


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

    if u == 0:
        # No interaction, return zero self-energy
        report("Skipping...")
        report("")
        return

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

    elif solver_type == "cthyb":
        from triqs_cthyb.tail_fit import tail_fit

        from .solvers.cthyb import solve_cthyb

        solver = solve_cthyb(params, u, e_onsite, delta)

        # Write results back to temporary file
        mpi.barrier()
        with HDFArchive(str(tmp_file), "a") as ar:
            ar["solver"] = solver
            ar["sigma_dmft"] = solver.Sigma_iw

        if params.solver_params.tail_fit:
            report("Fitting tail of Σ(z)...")
            if params.solver_params.fit_min_n == 0:
                fit_min_n = int(0.5 * params.n_iw)
            else:
                fit_min_n = params.solver_params.fit_min_n
            if params.solver_params.fit_max_n == 0:
                fit_max_n = params.n_iw
            else:
                fit_max_n = params.solver_params.fit_max_n
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
                fit_min_n,
                fit_max_n,
                fit_max_moment=fit_max_moment,
                fit_known_moments=sigma_moments,
            )

            with HDFArchive(str(tmp_file), "a") as ar:
                ar["sigma_dmft_raw"] = ar["sigma_dmft"]
                ar["sigma_dmft"] = sigma_fitted

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def solve_impurities_seq(
    params: InputParameters,
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

    # Write parameters and data to temporary files
    if mpi.is_master_node():
        for i, (cmpt, delt) in enumerate(delta):
            tmp_file = tmp_filepath.format(cmpt=cmpt)
            prepare_tmp_file(tmp_file, params, u[i], e_onsite[i], delta[cmpt])
    mpi.barrier()

    # --- Solve impurity problems -----
    for cmpt in sigma_dmft.indices:
        report(f"Solving component {cmpt}...")
        solve_impurity(tmp_filepath.format(cmpt=cmpt))
    mpi.barrier()
    report("Solvers done!")

    # ---- End solve ------------------

    # Load results back from temporary file
    for cmpt, sig in sigma_dmft:
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        report(f"Loading temporary file {tmp_file}...")
        with HDFArchive(str(tmp_file), "r") as ar:
            sig << ar[f"sigma_dmft"]  # noqa
        # Remove temporary file
        mpi.barrier()
        if mpi.is_master_node():
            Path(tmp_file).unlink(missing_ok=True)


def solve_impurities(
    params: InputParameters,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
    sigma_dmft: BlockGf,
    nproc: int,
    it: int = None,
    out_mode: str = "a",
    verbosity: int = 2,
    use_srun: bool = None,
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
    it : int, optional
        The current iteration number. Used for logging. The default is None.
    out_mode : str, optional
        The mode to open the log files. The default is "a".
    verbosity : int, optional
        The verbosity level. The default is 2.
    use_srun : bool, optional
        Whether to use the `srun` command to start the processes. If false, the `mpirun` command
        is used. The default is None, which uses the value of the global variable `USE_SRUN`.

    See Also
    --------
    solve_impurity : Solve the impurity problem using the parameters and data from a temporary file.
    """
    tmp_dir = Path(params.tmp_dir_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = str(tmp_dir / "tmp-{cmpt}.h5")
    stdout_filepath = str(Path(params.location_path) / "solver-{cmpt}.log")
    stderr_filepath = str(Path(params.location_path) / "solver-err-{cmpt}.log")
    if use_srun is None:
        use_srun = USE_SRUN

    if not mpi.is_master_node():
        return  # This method should only be called from the master node

    # Write parameters and data to temporary files
    for i, (cmpt, delt) in enumerate(delta):
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        prepare_tmp_file(tmp_file, params, u[i], e_onsite[i], delta[cmpt])

    # --- Solve impurity problems -----

    procs = list()
    executable = sys.executable  # Use current Python executable for subprocesses
    n = nproc / params.n_cmpt
    if n % 1 != 0:
        raise ValueError("Number of processes must be divisible by number of components.")
    # elif n > 4:
    #     raise ValueError("Number of processes per component must be less than 4.")
    n = max(1, int(n))

    if use_srun:
        base_cmd = ["srun", "--exact", "--exclusive", f"--ntasks={n}"]
    else:
        base_cmd = ["mpirun", "-n", str(n)] if n > 1 else list()

    for cmpt in sigma_dmft.indices:
        # Prepare tmp/output file paths
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        out_file = stdout_filepath.format(cmpt=cmpt)
        err_file = stderr_filepath.format(cmpt=cmpt)
        # Write header to output file
        write_header(out_file, it, out_mode)
        # write_header(err_file, it, out_mode)

        # Start process
        cmd = base_cmd + [executable, "-m", "model_dmft", "solve_impurity", tmp_file]
        if verbosity > 0:
            report("> " + " ".join(cmd))
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        procs.append((p, out_file, err_file))

    # Wait for all subprocesses to end and stream STDOUT to file and console
    while True:
        received = False
        for p, out_file, _ in procs:
            with open(out_file, "a") as f:
                line = p.stdout.readline().decode()
                if line:
                    line = line.strip()
                    if verbosity > 2:
                        report(f"[{p.pid}] " + line)
                    f.write(line + "\n")
                    received = True
        if not received:
            break

    # Wait to make sure all processes are finished
    for p, _, _ in procs:
        p.wait()

    # Check for errors
    for p, out_file, err_file in procs:
        err = p.stderr.read().decode()
        # with open(err_file, "a") as f:
        #     f.write(err)
        if p.returncode != 0:
            with open(out_file, "a") as f:
                f.write("#" * 100 + "\n")
                f.write(" STDERR\n")
                f.write("#" * 100 + "\n\n")
                f.write(err + "\n")
            raise SubprocessError(f"Error in process {p.pid}\n{err}")

    # ---- End solve ------------------

    # Load results back from temporary file
    for cmpt, sig in sigma_dmft:
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        with HDFArchive(str(tmp_file), "r") as ar:
            if "sigma_dmft" in ar:
                sig << ar["sigma_dmft"]  # noqa
        # Remove temporary file
        Path(tmp_file).unlink(missing_ok=True)


def anacont_pade(params: InputParameters, gf_iw: BlockGf) -> BlockGf:
    """Perform analytic continuation using Pade approximation.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    gf_iw : BlockGf, optional
        The input Green's function. If given, the Green's function is used for the continuation.
    """
    pade_params = params.pade_params
    mesh = pade_params.mesh
    kwargs = dict(n_points=pade_params.n_points, freq_offset=pade_params.freq_offset)

    gf_w = blockgf(mesh=mesh, names=params.spin_names, gf_struct=params.gf_struct, name="G_w")
    for name, g in gf_iw:
        gf_w[name].set_from_pade(g, **kwargs)
    return gf_w


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
            with HDFArchive(out_file, "r") as ar:
                sigma_cpa = ar["sigma_cpa"]
                g_coh = ar["g_coh"]
                g_cmpt = ar["g_cmpt"]

            report("Analytic continuation of G(iw) to G(ω) using Pade approximants...")
            g_coh_real = anacont_pade(params, g_coh)
            names, blocks = list(), list()
            for cmpt, g in g_cmpt:
                names.append(cmpt)
                blocks.append(anacont_pade(params, g))
            g_cmpt_real = blockgf(g_coh_real.mesh, names=names, blocks=blocks)

            report("Analytic continuation of Σ(iw) to Σ(ω) using Pade approximants...")
            sigma_cpa_real = anacont_pade(params, sigma_cpa)

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
    mu = params.mu

    e_onsite = eps + h_field * SIGMA - mu - u / 2  # Shape: ([N_cmpt, N_spin])

    # ---- COMPUTATIONAL PARAMETERS ----------------------------------------------------------------
    location = Path(params.location)
    out_file = params.output_path
    mesh = params.mesh
    n_loops = params.n_loops

    # DMFT parameters
    has_interaction = any(u)
    mixing_dmft = params.mixing_dmft
    # tol_dmft = params.tol_dmft

    # CPA parameters
    cpa_kwds = dict()
    cpa_kwds["method"] = params.method_cpa
    cpa_kwds["tol"] = params.tol_cpa
    cpa_kwds["maxiter"] = params.maxiter_cpa
    cpa_kwds["verbosity"] = params.verbosity_cpa

    # Check compatibility of solver
    solver_type = params.solver
    solver_params = params.solver_params
    if solver_type:
        supported = get_supported_solvers()
        if solver_type not in supported:
            raise ValueError(f"Solver {solver_type} is not supported. Use one of {supported}.")

        if params.is_real_mesh:
            if solver_type not in ("ftps",):
                raise ValueError(f"Solver {solver_type} is not compatible with real mesh.")
        else:
            if solver_type not in ("cthyb",):
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
    sigma_dmft = cpa.initialize_gf_cmpt(sigma_cpa, cmpts=len(conc))
    for i, (cmpt, sigma) in enumerate(sigma_dmft):
        for spin, sig in sigma:
            sig << u[i] / 2

    # Initialize effective onsite energies as BlockGf
    eps = cpa.initalize_onsite_energy(sigma_cpa, conc, e_onsite)

    # ---- START OF COMPUTATION --------------------------------------------------------------------

    print_params(params)
    report("")
    report(f"Starting job '{params.jobname}' at {start_time:%H:%M %d-%b-%y}")

    it_prev = 0
    if mpi.is_master_node():
        location.mkdir(parents=True, exist_ok=True)
        it_prev, sigma_dmft_prev, sigma_cpa_prev = load_state(params)
        if it_prev > 0:
            if sigma_dmft_prev:
                for cmpt, sig in sigma_dmft_prev:
                    # noinspection PyStatementEffect
                    sigma_dmft[cmpt] << sig
            if sigma_cpa_prev is not None:
                sigma_cpa << sigma_cpa_prev

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

    gf_coh_old = None
    sig_coh_old = None
    occ_old = None

    # Start iterations
    try:
        for it in range(it_prev + 1, n_loops + 1):
            mpi.barrier()
            report("")
            report_header(f"Iteration {it} / {n_loops}", width=100, char="=")
            report("")

            # Calculate local (component) Green's functions
            report("Computing component Green's function G_i(z)...")
            eps_eff = eps + sigma_dmft
            g_cmpt = cpa.gf_component(ht, sigma_cpa, conc, eps_eff, eta=eta, scale=False)

            # Solve impurity problems
            if any(u):
                sigma_old = sigma_dmft.copy()

                # Compute bath hybridization function
                report("Computing bath hybdridization function Δ(z)...")
                delta = hybridization(g_cmpt, eps, sigma_dmft, eta=eta)

                if mpi.is_master_node():
                    with HDFArchive(out_file, "a") as ar:
                        # ar["sigma_dmft"] = sigma_dmft
                        # ar[f"error_dmft"] = err_dmft
                        ar["delta"] = delta
                # Solve impurity problems to obtain new sigma_dmft
                kwargs = dict(
                    params=params, u=u, e_onsite=e_onsite, delta=delta, sigma_dmft=sigma_dmft
                )
                if mpi.is_master_node():
                    report("Solving for impurity self-energies Σ_i(z)...")
                    report("")

                if n_procs > 1:
                    report("Starting processes...")
                    solve_impurities(**kwargs, nproc=n_procs, it=it, use_srun=False)
                    report("Processes done!")
                else:
                    solve_impurities_seq(**kwargs)

                # Symmetrize DMFT self-energies
                if params.symmetrize:
                    for cmpt, sig in sigma_dmft:
                        symmetrize_gf(sig)

                # Apply mixing
                for cmpt, sig in sigma_dmft:
                    apply_mixing(sigma_old[cmpt], sig, mixing_dmft)

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
            sigma_cpa << cpa.solve_cpa(ht, sigma_cpa, conc, eps_eff, eta=eta, **cpa_kwds)

            # Symmetrize CPA self-energy
            if params.symmetrize:
                symmetrize_gf(sigma_cpa)

            report("")
            if mpi.is_master_node():
                report("Computing coherent Green's function...")
                g_coh = cpa.gf_coherent(ht, sigma_cpa, eta)

                report("Computing component Green's function...")
                g_cmpt = cpa.gf_component(ht, sigma_cpa, conc, eps_eff, eta, scale=False)

                # Compute occupation numbers
                density = g_coh.density()
                occ_up = density[up][0, 0].real
                occ_dn = density[dn][0, 0].real
                occ = g_coh.total_density().real
                report(f"Occupation: total={occ:.4f} up={occ_up:.4f} dn={occ_dn:.4f}")

                # Check convergence
                if gf_coh_old is not None:
                    err_g_coh = check_convergence(gf_coh_old, g_coh, relative=True)  # type: ignore
                else:
                    err_g_coh = 1.0
                if sig_coh_old is not None:
                    err_sigma = check_convergence(sig_coh_old, sigma_cpa, relative=True)  # type: ignore
                else:
                    err_sigma = 1.0
                err_occ = abs(occ - occ_old) / occ_old if occ_old is not None else 1.0

                sig_coh_old = sigma_cpa.copy()
                gf_coh_old = g_coh.copy()
                occ_old = occ

                report(f"Writing results of iteration {it} to {out_file}")
                with HDFArchive(out_file, "a") as ar:
                    ar["it"] = it
                    ar["sigma_cpa"] = sigma_cpa
                    ar["g_coh"] = g_coh
                    ar["g_cmpt"] = g_cmpt
                    ar["occ"] = occ
                    ar["err_g"] = err_g_coh
                    ar["err_sigma"] = err_sigma
                    ar["err_occ"] = err_occ

                # Run postprocessing
                postprocess(params, out_file)

                # Set the latest datasets and remove previous iterations if not needed
                update_dataset(out_file, keep_iter=params.store_iter)

                report("")
                s = "Iteration: {it:>2} Error-G: {g:.10f} Error-Σ: {sig:.10f} Error-n: {occ:.10f}"
                report(s.format(it=it, g=err_g_coh, sig=err_sigma, occ=err_occ))

                if not has_interaction:
                    report("No interaction, skipping further iterations.")
                    break
                if params.stol and err_sigma < params.stol:
                    now = datetime.now()
                    report("")
                    report(f"Σ converged in {it} iterations at {now:%H:%M %d-%b-%y}")
                    break
                if params.gtol and err_g_coh < params.gtol:
                    now = datetime.now()
                    report("")
                    report(f"G converged in {it} iterations at {now:%H:%M %d-%b-%y}")
                    break
                if params.occ_tol and err_occ < params.occ_tol:
                    now = datetime.now()
                    report("")
                    report(f"Occupation converged in {it} iterations at {now:%H:%M %d-%b-%y}")
                    break

        # Postprocessing
        postprocess(params, out_file)

        # Write output files as plain text
        report("Writing output files...")
        write_out_files(params)

        end_time = datetime.now()
        run_duration = end_time - start_time
        report("")
        report(f"Finished job '{params.jobname}' at {end_time:%H:%M %d-%b-%y}")
        report(f"Total time: {run_duration}")

    finally:
        try:
            pass
            # shutil.rmtree(params.tmp_dir)
        except Exception as e:
            report(f"Error: {e}")
