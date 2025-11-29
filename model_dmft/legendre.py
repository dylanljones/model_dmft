# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-19

from typing import Optional

import numpy as np
from triqs.gf import BlockGf
from triqs.gf.tools import fit_legendre as _fit_legendre

from .utility import toarray


def apply_legendre_filter(g_tau: BlockGf, order: int = 100, g_l_cut: float = 1e-19) -> BlockGf:
    """Filter binned imaginary time Green's function using a Legendre filter.

    Parameters
    ----------
    g_tau : TRIQS imaginary time Block Green's function
    order : int
        Legendre expansion order in the filter
    g_l_cut : float
        Legendre coefficient cut-off

    Returns
    -------
    g_l : TRIQS Legendre Block Green's function
        Fitted Green's function on a Legendre mesh
    """
    l_g_l = []
    for _, g in g_tau:
        g_l = _fit_legendre(g, order=order)
        g_l.data[:] *= np.abs(g_l.data) > g_l_cut
        g_l.enforce_discontinuity(np.identity(g.target_shape[0]))
        l_g_l.append(g_l)
    g_l = BlockGf(name_list=list(g_tau.indices), block_list=l_g_l, name="G_l")
    return g_l


def get_optimal_nl(g_l: BlockGf, thresh: float = 1e-2) -> int:
    """Get optimal Legendre expansion order for filtering.

    Parameters
    ----------
    g_l : TRIQS Legendre Block Green's function
        Green's function on a Legendre mesh
    thresh : float
        The threshold for determining the optimal order

    Returns
    -------
    nl : int
        Optimal Legendre expansion order, or `0` if no order is found.
    """
    idx_min = 100000000
    for cmpt, block in g_l:
        for spin, gf in block:
            data = toarray(gf)[:, 0, 0]
            data = np.abs(data[::2])
            indices = np.where(data < thresh)[0]
            if len(indices) == 0:
                return 0
            idx_thresh = indices[0]
            new_idx = int(idx_thresh) * 2
            idx_min = min(idx_min, new_idx)
    return idx_min + 1


def check_nl(g_l: BlockGf, n_l: int, thresh: float = 1e-3) -> Optional[int]:
    """Check the current Legendre Green's function and return new order if necessary.

    Parameters
    ----------
    g_l : TRIQS Legendre Block Green's function
        Green's function on a Legendre mesh
    n_l : int
        The current Legendre expansion order
    thresh : float
        The threshold for determining the optimal order

    Returns
    -------
    nl : int
        Optimal Legendre expansion order, or `None` if no change is necessary.
    """
    count_below = 0
    for cmpt, block in g_l:
        for spin, gf in block:
            data = toarray(gf)[:, 0, 0]
            data = np.abs(data[::2])
            # Count number of points below threshold
            indices = np.where(data < thresh)[0]
            count_below = max(count_below, len(indices))
    if count_below == 0:
        # If nl is to low, we increase it by 2
        new_nl = n_l + 2
        return new_nl
    elif count_below > 1:
        # If we have more than one point below threshold, we increase nl via threshold
        new_nl = get_optimal_nl(g_l, thresh)
        # make sure that we don't go below 2 or change by one
        new_nl = max(2, new_nl)
        if abs(new_nl - n_l) > 1:
            return new_nl
    return None
