# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2026-02-12

from typing import List

import numpy as np
from triqs.gf import BlockGf, Gf, MeshReFreq

# noinspection PyUnresolvedReferences
try:
    from triqs_maxent import (
        DataDefaultModel,
        DataOmegaMesh,
        HyperbolicOmegaMesh,
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

from .utility import GfLike, get_block, iter_blockgf, toarray

ALPHA_MESHES = {
    "linear": LinearAlphaMesh,
    "log": LogAlphaMesh,
}
OMEGA_MESHES = {
    "linear": LinearOmegaMesh,
    "hyperbolic": HyperbolicOmegaMesh,
    "lorentzian": LorentzianOmegaMesh,
}


def init_maxent(
    w_range: List[float],
    n_w: int = 201,
    n_alpha: int = 60,
    alpha_min: float = 0.001,
    alpha_max: float = 1000,
    alpha_mesh_type: str = "log",
    w_mesh_type: str = "hyperbolic",
    cost_function: str = "bryan",
    probability: str = "normal",
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
    AlphaMesh = ALPHA_MESHES[alpha_mesh_type]  # noqa: N806
    OmegaMesh = OMEGA_MESHES[w_mesh_type]  # noqa: N806

    # Initialize the meshes
    alpha_mesh = AlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=n_alpha)
    omega_mesh = OmegaMesh(omega_min=w_range[0], omega_max=w_range[1], n_points=n_w)

    tm = TauMaxEnt(cost_function=cost_function, probability=probability)
    tm.alpha_mesh = alpha_mesh
    tm.omega = omega_mesh
    if verbose:
        tm.set_verbosity(VerbosityFlags.AlphaLoop)
    else:
        tm.set_verbosity(VerbosityFlags.Quiet)
    tm.interactive = False

    return tm


def run_maxent_iw(
    tm: TauMaxEnt,
    g_iw: Gf,
    n_tau: int = -1,
    err: float = 1e-2,
    g_w_dm: Gf = None,
) -> MaxEntResult:
    assert isinstance(g_iw.mesh, MeshReFreq), "Input Green's function must be MeshReFreq"
    assert g_iw.target_shape == (1, 1), "Input Green's function must have shape (1, 1)"

    if g_w_dm is not None:
        assert isinstance(g_w_dm.mesh, MeshReFreq), "Default model mesh must be MeshReFreq"
        assert g_w_dm.target_shape == (1, 1), "Default model must have shape (1, 1)"
        omega_mesh_dm = DataOmegaMesh(toarray(g_w_dm.mesh))  # type: ignore
        dm_data = -toarray(g_w_dm)[:, 0, 0].imag / np.pi
        tm.D = DataDefaultModel(dm_data, omega_mesh_dm, tm.omega)

    tm.set_error(err)
    tm.set_G_iw(g_iw, np_tau=n_tau)  # type: ignore
    result = tm.run()

    return result


def anacont_maxent(
    g_iw: BlockGf,
    w_range: List[float],
    n_w: int = 201,
    err: float = 1e-2,
    n_alpha: int = 60,
    n_tau: int = -1,
    alpha_min: float = 0.001,
    alpha_max: float = 1000,
    alpha_mesh_type: str = "log",
    w_mesh_type: str = "hyperbolic",
    cost_function: str = "bryan",
    probability: str = "normal",
    g_w_dm: GfLike = None,
    verbose: bool = True,
) -> dict[tuple, MaxEntResult]:
    tm = init_maxent(
        w_range, n_w, n_alpha, alpha_min, alpha_max, alpha_mesh_type, w_mesh_type, cost_function, probability, verbose
    )

    results = dict()
    for keys, g in iter_blockgf(g_iw):
        if g_w_dm is not None:
            g_dm = get_block(g_w_dm, *keys)
        else:
            g_dm = None
        result = run_maxent_iw(tm, g, n_tau, err, g_dm)
        results[keys] = result
    return results
