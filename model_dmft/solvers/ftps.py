# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-05

"""ForkTPS helper functions."""

# import logging
import shutil
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import ContextManager, Iterable, Union

import forktps
import numpy as np
import triqs.operators as ops
from forktps.BathFitting import BathFitter
from forktps.DiscreteBath import DiscretizeBath, SigmaDyson, TimeStepEstimation
from forktps.solver import DMRGParams, TevoParams
from forktps.solver_core import Bath, HInt, Hloc  # noqa
from triqs.gf import BlockGf
from triqs.plot.mpl_interface import plt
from triqs.utility import mpi

from ..input import FtpsSolverParams, InputParameters
from ..utility import GfLike, report, toarray

# logger = logging.getLogger(__name__)


# noinspection PyIncorrectDocstring
def construct_bath(
    delta: BlockGf,
    eta: float,
    nbath: int,
    bath_fit: bool = None,
    enforce_gap: list = None,
    ignore_weight: float = 0.0,
    **kwds,
) -> Bath:
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
    enforce_gap : list of float, optional
        Enforce a gap in the bath spectrum. Only used for a discrete bath when `bath_fit=False`.
        The default is None.
    ignore_weight : float, optional
        Ignore bath states with weight below this threshold. Only used when fitting the bath when
        `bath_fit=True`. The default is 0.0.
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
        return DiscretizeBath(delta, Nb=nbath, gap=enforce_gap, **kwds)

    # Try to fit bath hybridization function without fixing nbath
    bath = BathFitter(Nb=None).FitBath(delta, eta=eta, ignoreWeight=ignore_weight, **kwds)
    if bath.N // bath.NArms > nbath:
        # If the number of bath sites is too large, use given nbath
        bath = BathFitter(Nb=nbath).FitBath(delta, eta=eta, ignoreWeight=ignore_weight, **kwds)

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


def make_positive_definite(g: GfLike, eps: float = 0.0) -> GfLike:
    """Make the input positive definite."""
    if isinstance(g, BlockGf):
        for name, gf in g:
            make_positive_definite(gf, eps)
        return g

    for orb, w in product(range(g.target_shape[0]), g.mesh):
        if g[orb, orb][w].imag > eps:
            g[orb, orb][w] = g[orb, orb][w].real + 0.0j
    return g


def solve_ftps(
    params: InputParameters, u: np.ndarray, e_onsite: np.ndarray, delta: BlockGf
) -> forktps.Solver:
    gf_struct = params.gf_struct
    up, dn = params.spin_names

    mesh = delta.mesh
    solve_params: FtpsSolverParams = params.solver_params

    # Parameters for sector search
    sector = DMRGParams(maxmI=50, maxmIB=50, maxmB=50, tw=1e-10, nmax=5, sweeps=5)

    # Parameters for DMRG
    dmrg = DMRGParams(
        sweeps=solve_params.sweeps,
        prep_imagTevo=True,
        prep_method="TEBD",
        prep_time_steps=5,
        napph=2,
        maxm=solve_params.dmrg_maxm or solve_params.maxm,
        maxmI=solve_params.dmrg_maxmI or solve_params.maxmI,
        maxmIB=solve_params.dmrg_maxmIB or solve_params.maxmIB,
        maxmB=solve_params.dmrg_maxmB or solve_params.maxmB,
        tw=solve_params.dmrg_tw or solve_params.tw,
        nmax=solve_params.dmrg_nmax or solve_params.nmax,
    )

    # Parameters for time evolution
    time_steps = solve_params.time_steps
    if time_steps is None:
        time_steps = 1  # dummy, use TimeStepEstimation to estimate later
        use_estimator = True
    else:
        use_estimator = False

    tevo = TevoParams(
        dt=solve_params.dt,
        time_steps=time_steps,
        method=solve_params.method,
        maxm=solve_params.maxm,
        maxmI=solve_params.maxmI,
        maxmIB=solve_params.maxmIB,
        maxmB=solve_params.maxmB,
        tw=solve_params.tw,
        nmax=solve_params.nmax,
    )

    # Construct bath
    report("Constructing bath.")

    # Ensure delta is positive definite
    delta = make_positive_definite(delta, eps=0.0)

    bath = construct_bath(
        delta,
        params.eta,
        solve_params.n_bath,
        bath_fit=solve_params.bath_fit,
        ignore_weight=solve_params.ignore_weight,
    )
    if not check_bath(bath, delta, params.eta, plot=False):
        raise ValueError("Bath construction failed!")

    # Construct local and interaction Hamiltonian
    report("")
    report("")
    report("Initializing FTPS solver....")
    hloc = Hloc(gf_struct)
    hloc.Fill(up, [[e_onsite[0]]])
    hloc.Fill(dn, [[e_onsite[1]]])
    hint = HInt(u=u, j=0.0, up=0.0, dd=True)

    # Initialize solver
    solver = forktps.Solver(gf_struct, mesh.omega_min, mesh.omega_max, len(mesh))
    solver.b = bath  # Add bath to solver
    solver.e0 = hloc  # Add local Hamiltonian to solver

    # calculate time_steps
    if use_estimator:
        mpi.report("Estimating time steps...")
        time_steps = TimeStepEstimation(bath, eta=params.eta, dt=solve_params.dt)
        s = "TimeStepEstimation returned {} with given bath, 'eta' = {} and 'dt' = {}"
        mpi.report(s.format(time_steps, params.eta, solve_params.dt))
        tevo.time_steps = time_steps

    solve_kwds = {
        "eta": params.eta,
        "tevo": tevo,
        "params_GS": dmrg,
        "params_partSector": sector,
        "measurements": [ops.n(up, 0), ops.n(dn, 0)],  # Only measure impurity Gfs
    }
    tmp_dir = params.tmp_dir_path
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    if not tmp_dir.endswith("/"):
        tmp_dir += "/"
    solve_kwds["state_storage"] = tmp_dir

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

    return solver
