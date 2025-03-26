# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-20

from typing import List

import numpy as np
from triqs.gf import BlockGf, MeshImTime, MeshReFreq

from .input import InputParameters
from .utility import blockgf, report


def anacont_pade(
    gf_iw: BlockGf, w_range: List[float], n_w: int, n_points: int, eta: float = 1e-3
) -> BlockGf:
    """Perform analytic continuation using Pade approximation.

    Parameters
    ----------
    gf_iw : BlockGf
        The input Green's function. If given, the Green's function is used for the continuation.
    w_range : array_like
        The frequency range to evaluate the Green's function
    n_w : int
        The number of frequency points to evaluate.
    n_points : int
        The number of frequency points to evaluate.
    eta : float
        The imaginary broadening.
    """
    kwargs = dict(n_points=n_points, freq_offset=eta)
    names = list(gf_iw.indices)
    mesh = MeshReFreq(w_range, n_w)

    min_iw = gf_iw.mesh(0).value.imag
    if eta > min_iw:
        report("Warning: eta is larger than the minimum Matsubara frequency.")

    gf_w = blockgf(mesh=mesh, names=names, target_gf=gf_iw, name="G_w")
    for name, g in gf_iw:
        gf_w[name].set_from_pade(g, **kwargs)
    return gf_w


def anacont_maxent(params: InputParameters, g_iw: BlockGf) -> tuple:
    from triqs_maxent import (
        HyperbolicOmegaMesh,
        LinearAlphaMesh,
        LinearOmegaMesh,
        LogAlphaMesh,
        LorentzianOmegaMesh,
        TauMaxEnt,
    )

    alpha_meshes = {
        "linear": LinearAlphaMesh,
        "log": LogAlphaMesh,
    }
    omega_meshes = {
        "linear": LinearOmegaMesh,
        "hyperbolic": HyperbolicOmegaMesh,
        "lorentzian": LorentzianOmegaMesh,
    }

    # Prepare G(τ) for MaxEnt
    maxent_params = params.maxent_params

    mesh = MeshImTime(beta=params.beta, n_tau=2501, S="Fermion")
    g_tau = blockgf(mesh, names=params.spin_names, gf_struct=params.gf_struct)
    for name, g in g_tau:
        g.set_from_fourier(g_iw[name])

    # Initialize MaxEnt
    tm = TauMaxEnt(cost_function=maxent_params.cost_function, probability=maxent_params.probability)
    alpha_mesh = alpha_meshes[maxent_params.mesh_type_alpha]
    omega_mesh = omega_meshes[maxent_params.mesh_type_w]
    tm.alpha_mesh = alpha_mesh(*maxent_params.alpha_range, n_points=maxent_params.n_alpha)
    tm.omega = omega_mesh(*maxent_params.w_range, n_points=maxent_params.n_w)

    # Run MaxEnt
    analyzers = ["LineFitAnalyzer", "Chi2CurvatureAnalyzer"]
    a_outs = dict()
    for analyzer in analyzers:
        a_outs[analyzer] = np.zeros((len(tm.omega), 2))
    for i, (name, g) in enumerate(g_tau):
        tm.set_G_tau(g)
        tm.set_error(maxent_params.error)
        result = tm.run()
        for analyzer in analyzers:
            a_outs[analyzer][:, i] = result.get_A_out(analyzer)

    return tm.omega, a_outs
