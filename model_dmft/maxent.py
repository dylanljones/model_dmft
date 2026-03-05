# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2026-02-12

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from h5 import HDFArchive
from triqs.gf import Gf, MeshImFreq, MeshReFreq

# noinspection PyUnresolvedReferences
try:
    from triqs_maxent import (
        DataDefaultModel,
        DataOmegaMesh,
        HyperbolicOmegaMesh,
        # InversionSigmaContinuator,
        LinearAlphaMesh,
        LinearOmegaMesh,
        LogAlphaMesh,
        LorentzianOmegaMesh,
        MaxEntResult,
        TauMaxEnt,
        VerbosityFlags,
    )
except ImportError:
    raise ImportError("triqs_maxent is not installed. Please install it to use the postprocessing.")

from .folder import Folder
from .utility import get_block, iter_blockgf, report, toarray

ALPHA_MESHES = {
    "linear": LinearAlphaMesh,
    "log": LogAlphaMesh,
}
OMEGA_MESHES = {
    "linear": LinearOmegaMesh,
    "hyperbolic": HyperbolicOmegaMesh,
    "lorentzian": LorentzianOmegaMesh,
}


def _init_maxent_gf(
    w_range: List[float],
    n_w: int = None,
    n_alpha: int = None,
    alpha_min: float = None,
    alpha_max: float = None,
    alpha_mesh_type: str = None,
    w_mesh_type: str = None,
    cost_function: str = None,
    probability: str = None,
    verbose: bool = True,
) -> TauMaxEnt:
    """Initialize the MaxEnt solver with the specified parameters.

    Parameters
    ----------
    w_range : List[float]
        The range of frequencies for the omega mesh, specified as [w_min, w_max].
    n_w : int, optional
        The number of points in the omega mesh. Default is 201.
        This parameter has a significant impact on performance, as the MaxEnt solver scales with the
        number of frequencies.
    n_alpha : int, optional
        The number of points in the alpha mesh. Default is 60.
    alpha_min : float, optional
        The minimum value of the alpha parameter. Default is 0.001.
    alpha_max : float, optional
        The maximum value of the alpha parameter. Default is 1000.
    alpha_mesh_type : {'linear', 'log'}, optional
        The type of mesh to use for the alpha parameter. Default is 'log'.
    w_mesh_type : {'linear', 'hyperbolic', 'lorentzian'}, optional
        The type of mesh to use for the omega frequencies. Default is 'hyperbolic'.
    cost_function : {'bryan', 'classic'}, optional
        The cost function to use in the MaxEnt solver. Default is 'bryan'.
    probability : {'normal', 'laplace'}, optional
        The probability distribution to use in the MaxEnt solver. Default is 'normal'.
    verbose : bool, optional
        Whether to print verbose output during initialization. Default is True.
    """
    alpha_mesh_type = alpha_mesh_type or "log"
    w_mesh_type = w_mesh_type or "hyperbolic"
    cost_function = cost_function or "bryan"
    probability = probability or "normal"

    alpha_min = alpha_min if alpha_min is not None else 0.001
    alpha_max = alpha_max if alpha_max is not None else 1000
    n_alpha = n_alpha if n_alpha is not None else 60
    n_w = n_w if n_w is not None else 201

    AlphaMesh = ALPHA_MESHES[alpha_mesh_type]  # noqa: N806
    OmegaMesh = OMEGA_MESHES[w_mesh_type]  # noqa: N806

    tm = TauMaxEnt(cost_function=cost_function, probability=probability)
    tm.alpha_mesh = AlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha)
    tm.omega = OmegaMesh(omega_min=w_range[0], omega_max=w_range[1], n_points=n_w)
    if verbose:
        tm.set_verbosity(VerbosityFlags.AlphaLoop)
    else:
        tm.set_verbosity(VerbosityFlags.Quiet)
    tm.interactive = False
    return tm


def _run_maxent_gf(
    tm: TauMaxEnt,
    g_iw: Gf,
    n_tau: int = -1,
    err: float = 1e-2,
    g_dm_w: Optional[Gf] = None,
) -> MaxEntResult:
    assert isinstance(g_iw.mesh, MeshImFreq), "Input Green's function must be MeshImFreq"
    assert g_iw.target_shape == (1, 1), "Input Green's function must have shape (1, 1)"

    tm.set_G_iw(g_iw, np_tau=n_tau)  # type: ignore
    tm.set_error(err)
    if g_dm_w is not None:
        assert isinstance(g_dm_w.mesh, MeshReFreq), "Default model mesh must be MeshReFreq"
        assert g_dm_w.target_shape == (1, 1), "Default model must have shape (1, 1)"
        omega_mesh_dm = DataOmegaMesh(toarray(g_dm_w.mesh))  # type: ignore
        dm_data = -toarray(g_dm_w)[:, 0, 0].imag / np.pi
        tm.D = DataDefaultModel(dm_data, omega_mesh_dm, tm.omega)

    result = tm.run()
    return result


def _try_load_maxent(file: Union[str, Path], key: str, overwrite: bool) -> dict:
    if not Path(file).exists() or overwrite:
        return dict()
    try:
        with HDFArchive(str(file), "r") as ar:
            return ar[key]
    except KeyError:
        return dict()


def run_maxent_g_cmpt(
    folder: Folder,
    w_min: float,
    w_max: float,
    n_w: int = None,
    alpha_min: float = None,
    alpha_max: float = None,
    n_alpha: int = None,
    n_tau: int = -1,
    err: float = 1e-2,
    cost_function: str = None,
    probability: str = None,
    alpha_mesh_type: str = None,
    w_mesh_type: str = None,
    folder_dm: Optional[Folder] = None,
    dm_const: float = 0.5,
    output_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    with folder.archive("r") as ar:
        g_cmpt = ar["g_cmpt"]

    if folder_dm is not None:
        with folder_dm.archive("r") as ar_dm:
            g_cmpt_dm = ar_dm["g_cmpt"]
            g_cmpt_dm += -1j * np.pi * dm_const  # Add offset to spectral func (imaginary part)
    else:
        g_cmpt_dm = None

    if output_file is not None:
        output_file = str(output_file)
    else:
        output_file = folder.output_file

    tm = _init_maxent_gf(
        w_range=[w_min, w_max],
        n_w=n_w,
        n_alpha=n_alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_mesh_type=alpha_mesh_type,
        w_mesh_type=w_mesh_type,
        cost_function=cost_function,
        probability=probability,
        verbose=verbose,
    )

    maxent_data = _try_load_maxent(output_file, "maxent/g_cmpt", overwrite)

    for key_path, g in iter_blockgf(g_cmpt):
        assert g.target_shape == (1, 1), "Input Green's function must have shape (1, 1)"
        key = ", ".join([str(k) for k in key_path])
        if key in maxent_data:
            report(f"MaxEnt already computed for {key}, skipping.")
            continue

        report(f"Running MaxEnt for '{key}'")
        if g_cmpt_dm is not None:
            try:
                g_dm = get_block(g_cmpt_dm, *key_path)  # type: ignore
            except KeyError:
                raise KeyError(f"Default model block {key} not found in g_cmpt_dm")
        else:
            g_dm = None

        result = _run_maxent_gf(tm, g, n_tau, err, g_dm)

        with HDFArchive(str(output_file), "a") as ar:
            if "maxent" not in ar:
                ar.create_group("maxent")
            if "g_cmpt" not in ar["maxent"]:
                ar.create_group("maxent/g_cmpt")
            ar.create_group(f"maxent/g_cmpt/{key}")
            ar["maxent"]["g_cmpt"][key] = result.data

        if verbose:
            report()


def run_maxent_g_coh(
    folder: Folder,
    w_min: float,
    w_max: float,
    n_w: int = None,
    alpha_min: float = None,
    alpha_max: float = None,
    n_alpha: int = None,
    n_tau: int = -1,
    err: float = 1e-2,
    cost_function: str = None,
    probability: str = None,
    alpha_mesh_type: str = None,
    w_mesh_type: str = None,
    folder_dm: Optional[Folder] = None,
    dm_const: float = 0.5,
    output_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    with folder.archive("r") as ar:
        g_coh = ar["g_coh"]

    if folder_dm is not None:
        with folder_dm.archive("r") as ar_dm:
            g_coh_dm = ar_dm["g_coh"]
            g_coh_dm += -1j * np.pi * dm_const  # Add offset to spectral func (imaginary part)
    else:
        g_coh_dm = None

    if output_file is not None:
        output_file = str(output_file)
    else:
        output_file = folder.output_file

    tm = _init_maxent_gf(
        w_range=[w_min, w_max],
        n_w=n_w,
        n_alpha=n_alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_mesh_type=alpha_mesh_type,
        w_mesh_type=w_mesh_type,
        cost_function=cost_function,
        probability=probability,
        verbose=verbose,
    )

    maxent_data = _try_load_maxent(output_file, "maxent/g_coh", overwrite)

    for key, g in g_coh:
        assert g.target_shape == (1, 1), "Input Green's function must have shape (1, 1)"
        if key in maxent_data:
            report(f"MaxEnt already computed for {key}, skipping.")
            continue

        report(f"Running MaxEnt for '{key}'")
        if g_coh_dm is not None:
            try:
                g_dm = g_coh_dm[key]  # type: ignore
            except KeyError:
                raise KeyError(f"Default model block {key} not found in g_coh_dm")
        else:
            g_dm = None

        result = _run_maxent_gf(tm, g, n_tau, err, g_dm)

        with HDFArchive(str(output_file), "a") as ar:
            if "maxent" not in ar:
                ar.create_group("maxent")
            if "g_coh" not in ar["maxent"]:
                ar.create_group("maxent/g_coh")
            ar.create_group(f"maxent/g_coh/{key}")
            ar["maxent"]["g_coh"][key] = result.data

        if verbose:
            report()


def run_maxent_gfs(
    folder: Folder,
    w_min: float,
    w_max: float,
    n_w: int = None,
    alpha_min: float = None,
    alpha_max: float = None,
    n_alpha: int = None,
    n_tau: int = -1,
    err: float = 1e-2,
    cost_function: str = None,
    probability: str = None,
    alpha_mesh_type: str = None,
    w_mesh_type: str = None,
    folder_dm: Optional[Folder] = None,
    dm_const: float = 0.5,
    output_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    run_maxent_g_coh(
        folder=folder,
        w_min=w_min,
        w_max=w_max,
        n_w=n_w,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        n_alpha=n_alpha,
        n_tau=n_tau,
        err=err,
        cost_function=cost_function,
        probability=probability,
        alpha_mesh_type=alpha_mesh_type,
        w_mesh_type=w_mesh_type,
        folder_dm=folder_dm,
        dm_const=dm_const,
        output_file=output_file,
        overwrite=overwrite,
        verbose=verbose,
    )

    run_maxent_g_cmpt(
        folder=folder,
        w_min=w_min,
        w_max=w_max,
        n_w=n_w,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        n_alpha=n_alpha,
        n_tau=n_tau,
        err=err,
        cost_function=cost_function,
        probability=probability,
        alpha_mesh_type=alpha_mesh_type,
        w_mesh_type=w_mesh_type,
        folder_dm=folder_dm,
        dm_const=dm_const,
        output_file=output_file,
        overwrite=overwrite,
        verbose=verbose,
    )
