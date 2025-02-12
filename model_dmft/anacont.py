# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-10

import numpy as np
from triqs.gf import BlockGf, MeshImTime

from .input import InputParameters
from .utility import blockgf


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
