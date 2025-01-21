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
from triqs.gf import BlockGf
from triqs.utility import mpi

from . import cpa, dmft
from .functions import HilbertTransform
from .input import InputParameters
from .utility import (
    SIGMA,
    apply_mixing,
    blockgf,
    check_broadening,
    check_convergence,
    report,
)

USE_SRUN = Path.home().resolve().parts[1].lower() == "hpc"


def report_header(text: str, width: int, char: str = "-", fg: str = "") -> None:
    """Print a header with a given width."""
    report(char * width, fg=fg)
    report(f"{' ' + text + ' ':{char}^{width}}", fg=fg)
    report(char * width, fg=fg)


def write_header(file: str, it: int, mode: str) -> None:
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
            report(f"N warmup:       {solver.n_warmup_cycle}")
            report(f"N cycles:       {solver.n_cycles}")
            report(f"Length cycle:   {solver.length_cycle}")
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
        if params.conc != old_params.conc:
            raise ValueError("Concentration mismatch.")
        if params.u != old_params.u:
            raise ValueError("U mismatch.")
        if params.eps != old_params.eps:
            raise ValueError("Onsite energy mismatch.")
        if params.h_field != old_params.h_field:
            raise ValueError("H field mismatch.")
        if params.mu != old_params.mu:
            raise ValueError("Chemical potential mismatch.")
        if params.is_real_mesh:
            if params.w_range != old_params.w_range:
                raise ValueError("Frequency range mismatch.")
            if params.n_w != old_params.n_w:
                raise ValueError("Number of frequencies mismatch.")
            if params.eta != old_params.eta:
                raise ValueError("Broadening mismatch.")
        else:
            if params.beta != old_params.beta:
                raise ValueError("Inverse temperature mismatch.")
            if params.n_iw != old_params.n_iw:
                raise ValueError("Number of Matsubara frequencies mismatch.")

        # Solver parameters
        if params.solver != old_params.solver:
            raise ValueError("Solver mismatch.")

        if params.solver.lower() == "ftps":
            if params.solver_params.n_bath != old_params.solver_params.n_bath:
                raise ValueError("Number of bath sites mismatch.")
            if params.solver_params.time_steps != old_params.solver_params.time_steps:
                raise ValueError("Number of time steps mismatch.")
            if params.solver_params.dt != old_params.solver_params.dt:
                raise ValueError("Time step mismatch.")
            if params.solver_params.method != old_params.solver_params.method:
                raise ValueError("Solver method mismatch.")
            if params.solver_params.sweeps != old_params.solver_params.sweeps:
                raise ValueError("Number of sweeps mismatch.")
            if params.solver_params.tw != old_params.solver_params.tw:
                raise ValueError("Truncation weight mismatch.")
            if params.solver_params.maxm != old_params.solver_params.maxm:
                raise ValueError("Max bond dimension mismatch.")
            if params.solver_params.nmax != old_params.solver_params.nmax:
                raise ValueError("Max Krylov dimension mismatch.")


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
            with HDFArchive(out_file, "w"):
                pass
        else:
            report("Reading data...")
            with HDFArchive(out_file, "r") as ar:
                try:
                    check_compatible_input(out_file, params)
                    it_prev = ar["it"]
                    # key_sigma_dmft = f"sigma_dmft-{it_prev}"
                    # key_sigma_cpa = f"sigma_cpa-{it_prev}"
                    key_sigma_dmft = "sigma_dmft"
                    key_sigma_cpa = "sigma_cpa"
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
        ar[f"err_g-{it}"] = ar["err_g"]
        ar[f"err_sigma-{it}"] = ar["err_sigma"]
        if "sigma_dmft" in ar:
            ar[f"sigma_dmft-{it}"] = ar["sigma_dmft"]
            ar[f"delta-{it}"] = ar["delta"]


def write_out_files(params: InputParameters) -> None:
    """Write main quantities to plain text output files.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    """
    frmt = "%.16f"
    location = Path(params.location)
    archive_file = str(location / params.output)

    # Load data from archive
    with HDFArchive(archive_file, "r") as ar:
        g_coh = ar["g_coh"]
        g_cmpt = ar["g_cmpt"]
        sigma_coh = ar["sigma_cpa"]

    omega = np.array(list(g_coh.mesh.values()))

    # Write DOS file
    names, items = list(), list()
    names.append("omega")
    names.append("DOS(up)")
    names.append("DOS(dn)")
    items.append(omega)
    items.append(-g_coh["up"].data[:, 0, 0].imag / np.pi)
    items.append(-g_coh["dn"].data[:, 0, 0].imag / np.pi)
    # Components
    for name, g in g_cmpt:
        names.append(f"DOS({name}-up)")
        names.append(f"DOS({name}-dn)")
        items.append(-g["up"].data[:, 0, 0].imag / np.pi)
        items.append(-g["dn"].data[:, 0, 0].imag / np.pi)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / "dos.dat", data, header=header, fmt=frmt, delimiter="  ")

    # Write coherent self-energy file
    names = ["omega", "Re SIG(up)", "Im SIG(up)", "Re SIG(dn)", "Im SIG(dn)"]
    items = list()
    items.append(omega)
    items.append(sigma_coh["up"].data[:, 0, 0].real)
    items.append(sigma_coh["up"].data[:, 0, 0].imag)
    items.append(sigma_coh["dn"].data[:, 0, 0].real)
    items.append(sigma_coh["dn"].data[:, 0, 0].imag)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / "sigma_coh.dat", data, header=header, fmt=frmt, delimiter="  ")


def solve_impurities_seq(
    params: InputParameters,
    u: np.ndarray,
    e_onsite: np.ndarray,
    delta: BlockGf,
    sigma_dmft: BlockGf,
) -> None:
    """Solve the impurity problems sequentially."""
    tmp_dir = Path(params.tmp_dir_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = str(tmp_dir / "tmp-{cmpt}.h5")

    # Write parameters and data to temporary files
    if mpi.is_master_node():
        # report("Solving for impurity self-energies Σ_i(ω)...")
        # report("")

        for i, (cmpt, delt) in enumerate(delta):
            tmp_file = tmp_filepath.format(cmpt=cmpt)
            report(f"Writing temporary file {tmp_file}...")
            dmft.prepare_tmp_file(tmp_file, params, u[i], e_onsite[i], delta[cmpt])
    mpi.barrier()

    # --- Solve impurity problems -----
    for cmpt in sigma_dmft.indices:
        report(f"Solving component {cmpt}...")
        dmft.solve_impurity(tmp_filepath.format(cmpt=cmpt))
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
    """Solve the impurity problems using MPI."""
    tmp_dir = Path(params.tmp_dir_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filepath = str(tmp_dir / "tmp-{cmpt}.h5")
    stdout_filepath = str(Path(params.location_path) / "solver-{cmpt}.log")
    stderr_filepath = str(Path(params.location_path) / "solver-err-{cmpt}.log")
    if use_srun is None:
        use_srun = USE_SRUN

    if not mpi.is_master_node():
        return  # This method should only be called from the master node

    # if verbosity > 0:
    #     report("Solving for impurity self-energies Σ_i(ω)...")
    #     report("")

    # Write parameters and data to temporary files
    for i, (cmpt, delt) in enumerate(delta):
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        # if verbosity > 1:
        #    report(f"Writing temporary file {tmp_file}...")
        dmft.prepare_tmp_file(tmp_file, params, u[i], e_onsite[i], delta[cmpt])

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

    # if verbosity > 0:
    #     report("")

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

    # if verbosity > 0:
    #     report("Solvers done!")

    # ---- End solve ------------------

    # Load results back from temporary file
    for cmpt, sig in sigma_dmft:
        tmp_file = tmp_filepath.format(cmpt=cmpt)
        # if verbosity > 1:
        #     report(f"Loading temporary file {tmp_file}...")
        with HDFArchive(str(tmp_file), "r") as ar:
            sig << ar[f"sigma_dmft"]  # noqa
        # Remove temporary file
        Path(tmp_file).unlink(missing_ok=True)


def solve(params: InputParameters, n_procs: int = 0) -> None:
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

    # Start iterations
    try:
        for it in range(it_prev + 1, n_loops + 1):
            mpi.barrier()
            report("")
            report_header(f"Iteration {it} / {n_loops}", width=100, char="=")
            report("")

            # Calculate local (component) Green's functions
            report("Computing component Green's function G_i(ω)...")
            eps_eff = eps + sigma_dmft
            g_cmpt = cpa.gf_component(ht, sigma_cpa, conc, eps_eff, eta=eta, scale=False)

            # Update data of iteration
            # if mpi.is_master_node():
            #    with HDFArchive(out_file, "a") as ar:
            #         ar["g_cmpt"] = g_cmpt

            # Solve impurity problems
            if any(u):
                sigma_old = sigma_dmft.copy()

                # Compute bath hybridization function
                report("Computing bath hybdridization function Δ(ω)...")
                delta = dmft.hybridization(g_cmpt, eps, sigma_dmft, eta=eta)

                if mpi.is_master_node():
                    with HDFArchive(out_file, "a") as ar:
                        ar["sigma_dmft"] = sigma_dmft
                        # ar[f"error_dmft"] = err_dmft
                        ar["delta"] = delta
                # Solve impurity problems to obtain new sigma_dmft
                kwargs = dict(
                    params=params, u=u, e_onsite=e_onsite, delta=delta, sigma_dmft=sigma_dmft
                )
                if mpi.is_master_node():
                    report("Solving for impurity self-energies Σ_i(ω)...")
                    report("")

                if n_procs > 1:
                    report("Starting processes...")
                    solve_impurities(**kwargs, nproc=n_procs, it=it, use_srun=False)
                    report("Processes done!")
                else:
                    solve_impurities_seq(**kwargs)

                # Apply mixing and compute errors
                # err_dmft = dict()
                for cmpt, sig in sigma_dmft:
                    apply_mixing(sigma_old[cmpt], sig, mixing_dmft)
                    # err_dmft[cmpt] = check_convergence(sigma_old[cmpt], sig)
                # report("Error DMFT: " + ", ".join([f"{c}: {e:.2e}" for c, e in err_dmft.items()]))

                # Update data of iteration
                if mpi.is_master_node():
                    with HDFArchive(out_file, "a") as ar:
                        ar["sigma_dmft"] = sigma_dmft
                        # ar[f"error_dmft"] = err_dmft
                        ar["delta"] = delta
            mpi.barrier()
            report("")

            # Solve CPA self-consistent equations and update coherent potential (sigm_cpa)
            # sigma_cpa_old = sigma_cpa.copy()
            eps_eff = eps + sigma_dmft
            sigma_cpa << cpa.solve_cpa(ht, sigma_cpa, conc, eps_eff, eta=eta, **cpa_kwds)

            # report("")
            # report("=" * 100)
            report("")
            if mpi.is_master_node():
                report("Computing coherent Green's function...")
                g_coh = cpa.gf_coherent(ht, sigma_cpa, eta)

                report("Computing component Green's function...")
                g_cmpt = cpa.gf_component(ht, sigma_cpa, conc, eps_eff, eta, scale=False)

                # Check convergence
                if gf_coh_old is not None:
                    err_g_coh = check_convergence(gf_coh_old, g_coh, relative=True)  # type: ignore
                else:
                    err_g_coh = 1.0
                if sig_coh_old is not None:
                    err_sigma = check_convergence(sig_coh_old, sigma_cpa, relative=True)  # type: ignore
                else:
                    err_sigma = 1.0
                sig_coh_old = sigma_cpa.copy()
                gf_coh_old = g_coh.copy()

                report(f"Writing results of iteration {it} to {out_file}")
                with HDFArchive(out_file, "a") as ar:
                    ar["it"] = it
                    ar["sigma_cpa"] = sigma_cpa
                    ar["g_coh"] = g_coh
                    ar["g_cmpt"] = g_cmpt
                    ar["err_g"] = err_g_coh
                    ar["err_sigma"] = err_sigma

                # Set the latest datasets and remove previous iterations if not needed
                update_dataset(out_file, keep_iter=params.store_iter)

                report("")
                report(f"Iteration: {it:>2} Error-G: {err_g_coh:.10f} Error-Σ: {err_sigma:.10f}")

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

        # Write output files as plain text
        write_out_files(params)

        end_time = datetime.now()
        run_duration = end_time - start_time
        report("")
        report(f"Finished job '{params.jobname}' at {end_time:%H:%M %d-%b-%y}")
        report(f"Total time: {run_duration}")

    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        try:
            pass
            # shutil.rmtree(params.tmp_dir)
        except Exception as e:
            report(f"Error: {e}")
