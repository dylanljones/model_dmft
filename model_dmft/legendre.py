# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-19

import numpy as np
from triqs.gf import BlockGf
from triqs.gf.tools import fit_legendre as _fit_legendre


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
