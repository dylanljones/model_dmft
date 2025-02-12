# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-10

import numpy as np
from triqs.gf import BlockGf, MeshImTime
from triqs_maxent import (
    HyperbolicOmegaMesh,
    LinearAlphaMesh,
    LinearOmegaMesh,
    LogAlphaMesh,
    LorentzianOmegaMesh,
    TauMaxEnt,
)

from .input import InputParameters
from .utility import blockgf

ALPHA_MESHES = {
    "linear": LinearAlphaMesh,
    "log": LogAlphaMesh,
}
OMEGA_MESHES = {
    "linear": LinearOmegaMesh,
    "hyperbolic": HyperbolicOmegaMesh,
    "lorentzian": LorentzianOmegaMesh,
}


def anacont_pade(params: InputParameters, gf: BlockGf = None, sigma: BlockGf = None) -> BlockGf:
    """Perform analytic continuation using Pade approximation.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    gf : BlockGf, optional
        The input Green's function. If given, the Green's function is used for the continuation.
        Either `gf` or `sigma` must be given.
    sigma : BlockGf, optional
        The input self energy. If given, the self energy is used for the continuation.
        Either `gf` or `sigma` must be given.
    """
    pade_params = params.pade_params
    mesh = pade_params.mesh
    kwargs = dict(n_points=pade_params.n_points, freq_offset=pade_params.freq_offset)
    if gf is not None:
        gf_w = blockgf(mesh=mesh, names=params.spin_names, gf_struct=params.gf_struct, name="G_w")
        for name, g in gf:
            gf_w[name].set_from_pade(g, **kwargs)
        return gf_w
    elif sigma is not None:
        raise NotImplementedError("Analytic continuation from self energy is not implemented yet.")
    else:
        raise ValueError("Either gf or sigma must be given.")


def anacont_maxent(params: InputParameters, g_iw: BlockGf) -> tuple:
    # Prepare G(τ) for MaxEnt
    maxent_params = params.maxent_params

    mesh = MeshImTime(beta=params.beta, n_tau=2501, S="Fermion")
    g_tau = blockgf(mesh, names=params.spin_names, gf_struct=params.gf_struct)
    for name, g in g_tau:
        g.set_from_fourier(g_iw[name])

    # Initialize MaxEnt
    tm = TauMaxEnt(cost_function=maxent_params.cost_function, probability=maxent_params.probability)
    alpha_mesh = ALPHA_MESHES[maxent_params.mesh_type_alpha]
    omega_mesh = OMEGA_MESHES[maxent_params.mesh_type_w]
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
