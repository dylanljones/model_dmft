# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-05

"""ForkTPS helper functions."""

# import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Iterable, Union

import forktps
import numpy as np
import triqs.operators as ops
from forktps.BathFitting import BathFitter
from forktps.DiscreteBath import DiscretizeBath, SigmaDyson
from forktps.solver import DMRGParams, TevoParams
from forktps.solver_core import Bath, HInt, Hloc  # noqa
from triqs.gf import BlockGf
from triqs.plot.mpl_interface import plt
from triqs.utility import mpi

from ..input import FtpsSolverParams, InputParameters
from ..utility import report, toarray

# logger = logging.getLogger(__name__)


# noinspection PyIncorrectDocstring
def construct_bath(delta: BlockGf, eta: float, nbath: int, bath_fit: bool = None, **kwds) -> Bath:
    """Construct the bath from the hybridization function Δ.

    Parameters
    ----------
    delta : BlockGf
        The hybridization function Δ for both spin blocks.
    eta : float
        The complex broadening.
    nbath : int
        The number of bath sites to use.
    bath_fit : bool, optional
        Whether to use a discrete bath or fit the bath to the hybridization function.
        If None, a discrete bath is used if the broadening ´eta´ is smaller than the
        mesh discretization.
    **kwds
        Additional keyword arguments passed to `DiscretizeBath` or `FitBath`.
        See Additional Parameters for more information.

    Additional Parameters
    ---------------------
    err : float, optional
        The error threshold for discretizing the bath. Default: 0.01.
    SO : bool, optional
        Whether to use spin-orbit coupling for discretizing the bath. Default: False.
    gap : list of float, optional
        Bounds for the gap in the discrete bath. Default: None
    outerInt : list, optional
        Bounds for the outermost interval of the discrete bath. Default: None
    PlaceAt : list, optional
        Place the discrete baths at the given energyies. Default: None
    step : float, optional
        The step size used for fitting. Default: None
    ignoreL : float, optional
        Ignore the L-th bath site. Default: None
    ignoreWeight : float, optional
        Ignore bath sites with weight below the given threshold. Default: None
    fixNb : bool, optional
        Fix the number of bath sites to the given value. Default: False
    """
    # Get mesh discretization
    dw = min(np.diff(toarray(delta.mesh)))

    if not bath_fit or (bath_fit is None and eta < dw):
        # Use a discrete bath
        if mpi.is_master_node():
            report(f"Using discrete bath with {nbath} sites.")
            # logger.info("Using discrete bath with %s bath sites.", nbath)
        return DiscretizeBath(delta, Nb=nbath, **kwds)

    # Try to fit bath hybridization function without fixing nbath
    bath = BathFitter(Nb=None).FitBath(delta, eta=eta, **kwds)
    if bath.N // bath.NArms > nbath:
        # If the number of bath sites is too large, use given nbath
        bath = BathFitter(Nb=nbath).FitBath(delta, eta=eta, **kwds)

    if mpi.is_master_node():
        report(f"Fitted bath using {bath.N} bath sites ({bath.NArms} arms).")
        # logger.info("Fitted bath using %s bath sites (%s arms).", bath.N, bath.NArms)
    return bath


def plot_hybrid_reconstruction(
    x: np.ndarray, ori: np.ndarray, rec: np.ndarray, errs: np.ndarray
) -> None:
    """Plot the original hybridization vs the reconstructed one."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex="all", figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.set(ylabel="Im Δ(ω)", xmargin=0)
    ax1.axhline(0, color="dimgrey", lw=0.8)
    ax1.axvline(0, color="dimgrey", lw=0.8)

    ax1.plot(x, -ori[0].imag, lw=1.5, color="k", label="Im Δ")
    ax1.plot(x, -rec[0].imag, lw=1.5, color="C0", ls="--", label=r"Im Δ$_{rec}$")
    ax1.plot(x, +ori[1].imag, lw=1.5, color="k")
    ax1.plot(x, +rec[1].imag, lw=1.5, color="C0", ls="--")

    ax1.plot(x, -ori[0].real, lw=1.5, color="k", label="Re Δ")
    ax1.plot(x, -rec[0].real, lw=1.5, color="C1", ls="--", label=r"Re Δ$_{rec}$")
    ax1.plot(x, +ori[1].real, lw=1.5, color="k")
    ax1.plot(x, +rec[1].real, lw=1.5, color="C1", ls="--")
    ax1.legend()

    ax2.axvline(0, color="dimgrey", lw=0.8)
    ax2.set(xlabel="ω", ylabel=r"|Δ - Δ$_{rec}$|", xmargin=0)
    ax2.plot(x, errs[0], lw=1.5, label="up")
    ax2.plot(x, errs[1], lw=1.5, label="dn")
    ax2.set_ylim(0, None)
    ax2.legend()
    fig.tight_layout()


def check_bath(
    bath: Bath,
    delta: BlockGf,
    eta: float,
    atol: float = None,
    rtol: float = 1.0,
    plot: bool = False,
) -> bool:
    """Check the accuracy of the bath model."""
    if not mpi.is_master_node():
        return True
    report("Checking bath model...")

    delta_rec = bath.reconstructDelta(w=delta.mesh, eta=eta)
    up, dn = list(delta.indices)

    # Convert to numpy array
    x = toarray(delta.mesh)
    rec = toarray(delta_rec)[..., 0, 0]
    ori = toarray(delta)[..., 0, 0]

    # Compute errors
    aerrs = np.abs(rec - ori)
    rerrs = aerrs / np.abs(ori)

    # Compute norms
    aerr = np.linalg.norm(aerrs, axis=1)
    rerr = np.linalg.norm(rerrs, axis=1) / np.linalg.norm(ori, axis=1)
    report(f"Δ_{up} reconstruction error: {aerr[0]:.10f} ({rerr[0]:.10f})")
    report(f"Δ_{dn} reconstruction error: {aerr[1]:.10f} ({rerr[1]:.10f})")
    # logger.info("Δ_%s reconstruction error: %18.10f (%.10f)", up, aerr[0], rerr[0])
    # logger.info("Δ_%s reconstruction error: %18.10f (%.10f)", dn, aerr[1], rerr[1])

    if plot:
        plot_hybrid_reconstruction(x, ori, rec, aerrs)
        plt.show()
    else:
        s = "{} error of bath hybridization reconstruction is too large!"
        if atol is not None and np.any(aerr > atol):
            report(s.format("Absolute"))
            # logger.warning(s.format("Absolute"))
            return False
        if rtol is not None and np.any(rerr > rtol):
            report(s.format("Relative"))
            # logger.warning(s.format("Relative"))
            return False

        report("Bath model is accurate.")
        # logger.info("Bath model is accurate.")
    return True


def _rmtree(path: Union[Path, str]) -> None:
    path = Path(path)
    # Try to delete the directory
    try:
        shutil.rmtree(path)
        return
    except PermissionError:
        pass

    success = True
    # remove all files
    for file in path.rglob("*"):
        if file.is_file():
            try:
                file.unlink()
            except PermissionError as e:
                success = False
                report(f"Could not delete file {file.relative_to(path)}: {e}")

    # remove all sub-directories
    for dir_ in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if dir_.is_dir():
            try:
                # Delete directory if it is empty
                if not list(dir_.iterdir()):
                    dir_.rmdir()
                else:
                    raise RuntimeError(f"Directory {dir_.relative_to(path)} is not empty")
            except PermissionError as e:
                success = False
                report(f"Could not delete directory {dir_.relative_to(path)}: {e}")

    # Try to remove the whole directory again
    try:
        shutil.rmtree(path)
        return
    except PermissionError:
        pass

    if success:
        try:
            path.rmdir()
        except PermissionError as e:
            report(f"Could not delete tmp-dir {path}: {e}")


def _iter_tmpdirs(root: Union[Path, str] = None) -> Iterable[Path]:
    root = Path.cwd() if root is None else Path(root)
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("MPS"):
            yield path


def rm_tmpdirs(root: Union[Path, str] = None) -> None:
    if not mpi.is_master_node():
        return
    try:
        tmpdirs = list(_iter_tmpdirs(root))
        if len(tmpdirs) == 0:
            return  # No tmp-dirs found
        report("Removing FTPS tmp-files...")
        for tmpdir in tmpdirs:
            _rmtree(tmpdir)
    except Exception as e:
        report("Unknown error while cleaning FTPS tmp-files:")
        report(e)


@contextmanager
def cleantmp(
    root: Union[Path, str] = None, pre: bool = False, post: bool = True
) -> ContextManager[None]:
    if pre:
        rm_tmpdirs(root)
    try:
        yield
    finally:
        if post:
            rm_tmpdirs()


def solve_ftps(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> BlockGf:
    gf_struct = params.gf_struct
    up, dn = params.spin_names

    mesh = delta.mesh
    solve_params: FtpsSolverParams = params.solver_params
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
        "measurements": [ops.n(up, 0), ops.n(dn, 0)],  # Only measure impurity Gfs
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
    hloc.Fill(up, [[e_onsite[0]]])
    hloc.Fill(dn, [[e_onsite[1]]])
    hint = HInt(u=u, j=0.0, up=0.0, dd=True)

    # Initialize solver
    solver = forktps.Solver(gf_struct, mesh.omega_min, mesh.omega_max, len(mesh))
    solver.b = bath  # Add bath to solver
    solver.e0 = hloc  # Add local Hamiltonian to solver

    # Run solver
    report("Solving impurity...")
    solver.solve(h_int=hint, **solve_kwds)  # type: ignore

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
