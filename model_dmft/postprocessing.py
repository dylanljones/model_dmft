# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-20

from typing import List

import numpy as np
from triqs.gf import BlockGf, Gf, MeshImTime, MeshReFreq

from .input import InputParameters
from .utility import GfLike, blockgf, report


def anacont_pade(gf_iw: GfLike, w_range: List[float], n_w: int, n_points: int, eta: float = 1e-3) -> GfLike:
    """Perform analytic continuation using Pade approximation.

    Parameters
    ----------
    gf_iw : Gf or BlockGf
        The input Green's function. Can be a Gf, BlockGf or nested BlockGf.
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

    mesh = MeshReFreq(w_range, n_w)

    min_iw = gf_iw.mesh(0).value.imag
    if eta > min_iw:
        report("Warning: eta is larger than the minimum Matsubara frequency.")

    def _pade(g_in: Gf, g_out: Gf) -> None:
        g_out.set_from_pade(g_in, **kwargs)

    if isinstance(gf_iw, Gf):
        gf_w = Gf(mesh=mesh, name="G_w")
        _pade(gf_iw, gf_w)
        return gf_w
    else:
        # Check if the input is a BlockGf or nested BlockGf
        if isinstance(gf_iw, BlockGf) and all(isinstance(g, Gf) for _, g in gf_iw):
            # Input is a BlockGf of Gfs
            names = list(gf_iw.indices)
            gf_w = blockgf(mesh=mesh, names=names, target_gf=gf_iw, name="G_w")
            for name, g in gf_iw:
                _pade(g, gf_w[name])
            return gf_w
        else:
            # Assume the input is a nested BlockGf and apply Pade to each Gf in the nested structure
            names_outer = list(gf_iw.indices)
            blocks_outer = list()
            for name_outer, block_outer in gf_iw:
                names = list(block_outer.indices)
                gf_inner = blockgf(mesh=mesh, names=names, target_gf=block_outer)
                for name, g in block_outer:
                    _pade(g, gf_inner[name])
                blocks_outer.append(gf_inner)
            return BlockGf(block_list=blocks_outer, name_list=names_outer, name="G_w")


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
