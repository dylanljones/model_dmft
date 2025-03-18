# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-18

from datetime import datetime
from typing import Tuple, Union

import numpy as np
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshImTime, MeshReFreq

from .input import InputParameters
from .utility import TIME_FRMT, GfLike, report, walk_block_paths


def max_difference(old: GfLike, new: GfLike, norm_temp: bool = None) -> float:
    """Calculate the maximum difference between two Green's functions using the Frobenius norm.

    The Frobenius norm is calculated over the grid points of the Green's function. The result is
    divided by `sqrt(beta)` for Matsubara frequencies, `sqrt(N)` for real frequencies and
    `sqrt(N/beta)` for imaginary time frequencies.

    Parameters
    ----------
    old : GfLike
        The old quantity (Gf or self energy).
    new : GfLike
        The new quantity (Gf or self energy).
    norm_temp : bool, optional
        If `True` the norm is divided by an additional `sqrt(beta)` to account for temperature
        scaling. The default is `True`.

    Returns
    -------
    float
        The maximum difference between the two Green's functions.
    """
    if isinstance(old, BlockGf):
        diff = 0.0
        for name, g in old:
            diff += max_difference(old[name], new[name], norm_temp)
        return diff

    assert old.mesh == new.mesh, "Meshes of inputs do not match."
    mesh = old.mesh
    n_points = len(mesh)

    # subtract the largest real value to make sure that G1-G2 falls off to 0
    if isinstance(mesh, MeshImFreq):
        offset = np.diag(np.diag(old.data[-1, :, :].real - new.data[-1, :, :].real))
    else:
        offset = 0.0

    # calculate norm over all axis but the first one which are frequencies
    norm_grid = np.linalg.norm(old.data - new.data - offset, axis=tuple(range(1, old.data.ndim)))

    # now calculate Frobenius norm over grid points
    if isinstance(mesh, MeshImFreq):
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(mesh.beta)
    elif isinstance(mesh, MeshImTime):
        norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(mesh.beta / n_points)
    elif isinstance(mesh, MeshReFreq):
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(n_points)
    else:
        raise NotImplementedError(f"Mesh type {type(mesh)} not supported.")

    if norm_temp is None:
        norm_temp = True
    if isinstance(mesh, (MeshImFreq, MeshImTime)) and norm_temp:
        norm = norm / np.sqrt(mesh.beta)

    return float(norm)


def norm_max_difference(old: GfLike, new: GfLike, relative: bool = True) -> float:
    if isinstance(old, Gf):
        error = float(np.linalg.norm(old.data - new.data))
        if relative:
            error /= np.linalg.norm(old.data)
    else:
        norms = list()
        max_norm = 0
        for keys in walk_block_paths(old):
            item_old = old
            item_new = new
            for key in keys:
                item_old = item_old[key]
                item_new = item_new[key]
            norms.append(np.linalg.norm(item_old.data - item_new.data))
            max_norm = max(max_norm, np.linalg.norm(item_old.data))
        error = max(norms)

        if relative:
            error /= max_norm
    return error


def calculate_convergences(
    g_new: BlockGf,
    sigma_new: BlockGf,
    occ_new: float,
    g_old: Union[BlockGf, None],
    sigma_old: Union[BlockGf, None],
    occ_old: Union[float, None],
    relative: bool = True,
) -> Tuple[float, float, float]:
    """Calculate the convergence metrics for Green's functions, self-energies and occupations.

    Parameters
    ----------
    g_new : BlockGf
        The new Green's function.
    sigma_new : BlockGf
        The new self-energy.
    occ_new : float
        The new occupation numbers.
    g_old : BlockGf or None
        The old Green's function.
    sigma_old : BlockGf or None
        The old self-energy.
    occ_old : float or None
        The old occupation numbers.
    relative : bool, optional
        If `True` the error is normalized by the old value. The default is `True`.

    Returns
    -------
    err_g: float
        The error in the Green's function.
    err_sigma: float
        The error in the self-energy.
    err_occ: float
        The error in the occupation numbers.
    """
    # Default errors
    err_default = 1.0 if relative else np.inf
    err_g, err_sigma, err_occ = err_default, err_default, err_default

    if g_old is not None:
        # err_g = max_difference(g_old, g_new, norm_temp)
        err_g = norm_max_difference(g_old, g_new, relative)

    if sigma_old is not None:
        # err_sigma = max_difference(sigma_old, sigma_new, norm_temp)
        err_sigma = norm_max_difference(sigma_old, sigma_new, relative)

    if occ_old is not None:
        err_occ = abs(occ_new - occ_old)
        if relative:
            err_occ /= occ_old

    return err_g, err_sigma, err_occ


def check_convergence(
    params: InputParameters, it: int, err_g: float, err_sigma: float, err_occ: float
) -> bool:
    """Check if the convergence criteria are met.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    it : int
        The current iteration number.
    err_g : float
        The error in the Green's function.
    err_sigma : float
        The error in the self-energy.
    err_occ : float
        The error in the occupation numbers.

    Returns
    -------
    bool
        `True` if all convergence criteria are met, `False` otherwise.
    """
    if params.stol and err_sigma < params.stol:
        now = datetime.now()
        report("")
        report(f"Σ converged in {it} iterations at {now:{TIME_FRMT}}")
        report("")
        return True
    if params.gtol and err_g < params.gtol:
        now = datetime.now()
        report("")
        report(f"G converged in {it} iterations at {now:{TIME_FRMT}}")
        report("")
        return True
    if params.occ_tol and err_occ < params.occ_tol:
        now = datetime.now()
        report("")
        report(f"Occupation converged in {it} iterations at {now:{TIME_FRMT}}")
        report("")
        return True
    return False
