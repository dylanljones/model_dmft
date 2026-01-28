# Copyright (c) 2021-2024 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Original authors: Alexander Hampel, Harrison LaBollita, Nils Wentzell
# Updated by: Dylan Jones (2026)

import warnings
from typing import Union, Tuple, Dict

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from triqs.gf import (
    Gf,
    BlockGf,
    make_gf_dlr,
    make_gf_dlr_imfreq,
    make_gf_imfreq,
    make_hermitian,
    fit_gf_dlr,
    inverse,
    MeshDLRImFreq,
    MeshDLRImTime,
    MeshDLR
)
from triqs.utility import mpi

warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")


def minimize_dyson(
    g0_dlr: Union[Gf, BlockGf],
    g_dlr: Union[Gf, BlockGf],
    sigma_moments: Union[np.ndarray, Dict[str, np.ndarray]],
    method: str = "trust-constr",
    maxiter: int = 5000,
    gtol: float = 1e-32,
    xtol: float = 1e-100,
    finite_diff_rel_step: float = 1e-20,
    verbosity: int = 0,
    **kwargs,
) -> Tuple[Union[Gf, BlockGf], Union[np.ndarray, Dict[str, np.ndarray]], Union[float, Dict[str, float]]]:
    """
    Contrained Residual Minimization Dyson solver as described in https://arxiv.org/abs/2310.01266

    Defines the Dysons equation as an optimization problem:

        G - G0 - G0*Σ*G = 0

    and solves it using scipy.optimize.minimize using the DLR representation for the Green's functions.

    The solver optimizes only the dynamic part of the self-energy Σ_dyn(iν)= Σ(iν) - Σ_0.
    Here, Σ_0 is the Hartree shift. If provided the second moment Σ_1 is used as a non-linear constraint in the solver.

    The moments can be explicitly calculated in the impurity solver, see for example the `cthyb high frequency moments tutorial <https://triqs.github.io/cthyb/latest/guide/high_freq_moments.html>`_ .

    Alternatively the moments can be approximated by fitting the tail of the self-energy calculated via normal Dyson equation first:

    >>> S_iw = inverse(G0_iw) - inverse(G_iw)
    >>> tail, err = S_iw.fit_hermitian_tail()

    and then used as input for the Dyson solver:

    >>> s_iw_dlr, sigma_HF, res = minimize_dyson_crm(g0_dlr, g_dlr, sigma_moments=tail[0:1])

    Parameters
    ----------
    g0_dlr : triqs.gf.Gf or triqs.gf.BlockGf
        Non-interacting Green's function defined on a DLR, DLRImTime, or DLRImFreq mesh
    g_dlr : triqs.gf.Gf or triqs.gf.BlockGf
        Interacting Green's function defined on a DLR, DLRImTime, or DLRImFreq mesh
    sigma_moments : npndarray or dict of np.ndarray
        Moments of Σ. The first moment is the Hartree shift, i.e. the constant part of Σ.
        If provdided, use the second moment as a non-linear constraint for the Dyson solver.
    method : str, optional
        Optimization method, defaults to 'trust-constr'
        Note: For non-linear constraints this is one of the few available methods
    maxiter : int, optional
        Maximum number of iterations for the optimizer, defaults to 5000
    gtol : float, optional
        Gradient tolerance for the optimizer, defaults to 1e-32
    xtol : float, optional
        Step size tolerance for the optimizer, defaults to 1e-100
    finite_diff_rel_step : float, optional
        Relative step size for finite difference gradient approximation, defaults to 1e-20
    verbosity: int, optional
        Verbosity level for mpi reporting (0: warnings, 1: info, 2: full). Defaults to 0 (warnings)
    **kwargs
        Additional keyword arguments for scipy.optimize.minimize

    Returns
    -------
    Sigma_DLR : triqs.gf.Gf or triqs.gf.BlockGf
        optimized self-energy defined on a DLRImFreq mesh
    Sigma_0 : np.ndarray or dict of np.ndarray
        Hartree shift
    residual : float or dict of float
        L2 norm of residual (G-G₀-G₀ΣG)
    """

    # recursive call for BlockGf, could be MPI parallelized
    if isinstance(g_dlr, BlockGf) or isinstance(g0_dlr, BlockGf):
        assert isinstance(g_dlr, BlockGf) and isinstance(g0_dlr, BlockGf), "G0_dlr and G_dlr must be both Gf or BlockGf"
        assert list(g_dlr.indices).sort() == list(g0_dlr.indices).sort(), "G0_dlr and G_dlr must have the same block structure"
        block_list = []
        sig_hf = {}
        residuals = {}
        for block, gtau in g_dlr:
            if verbosity > 0:
                mpi.report(f"Block {block}...")
            sig_dlr, sig_hf, res = minimize_dyson(
                g0_dlr[block],
                g_dlr[block],
                sigma_moments[block],
                method,
                maxiter,
                gtol,
                xtol,
                finite_diff_rel_step,
                verbosity,
                **kwargs
            )
            block_list.append(sig_dlr)
            sig_hf[block] = sig_hf
            residuals[block] = res

        bgf_sigma_iw_fit = BlockGf(name_list=list(g_dlr.indices), block_list=block_list)

        return bgf_sigma_iw_fit, sig_hf, residuals

    options = dict(
        maxiter=maxiter,
        gtol=gtol,
        xtol=xtol,
        finite_diff_rel_step=finite_diff_rel_step,
        disp=verbosity > 1,
        **kwargs
    )

    # initial checks
    if len(sigma_moments) > 0:
        assert g_dlr.target_shape == g0_dlr.target_shape == sigma_moments.shape[1:], "number of orbs inconsistent across G, G0, and moments"
    else:
        raise ValueError("Provide self-energy moments for the Dyson solver as numpy.ndarray or dict of numpy.ndarray")

    # make sure we are working with matrix valued Green's functions
    if len(g_dlr.target_shape) == 0:
        g_dlr = Gf(mesh=g_dlr.mesh, data=g_dlr.data.reshape(-1, 1, 1))
        g0_dlr = Gf(mesh=g0_dlr.mesh, data=g0_dlr.data.reshape(-1, 1, 1))
        sigma_moments = sigma_moments.reshape(-1, 1, 1)
        scalar_output = True
    else:
        scalar_output = False

    # prepare meshes
    def to_dlr_imfreq(g):
        if isinstance(g.mesh, (MeshDLRImTime, MeshDLR)):
            return make_gf_dlr_imfreq(g)
        elif isinstance(g.mesh, MeshDLRImFreq):
            return g
        else:
            raise ValueError(
                "minimize_dyson input Green functions must be defined on "
                f"MeshDLRImFreq, MeshDLRImTime, or MeshDLR, but got {g.mesh}"
            )
    g0_iwaa  = to_dlr_imfreq(g0_dlr)
    g_iwaa   = to_dlr_imfreq(g_dlr)
    assert g0_iwaa.mesh == g_iwaa.mesh, f'G0_dlr and G_dlr have incompatible dlr meshes {g0_iwaa.mesh} and {g_iwaa.mesh}'
    mesh_iw  = g_iwaa.mesh

    def flatten(arr):
        """Gf / mat -> vector conversion."""
        return arr.flatten().view(float)

    def unflatten(vec):
        """vector > Gf / mat conversion."""
        return vec.view(complex).reshape(g_dlr.data.shape)

    # setup constraints
    if len(sigma_moments) == 1:
        constraints = ()
    else:  # len(Sigma_moments) >= 2, use only the second moment

        def constraint_func(x):
            """Constraint condition: ∑σk =  Σ_1."""
            temp = Gf(mesh=mesh_iw, data=unflatten(x))
            sig = make_gf_dlr(temp)
            mat = sig.data.sum(axis=0)
            vec = flatten(mat)
            return vec

        bound = flatten(sigma_moments[1])
        constraints = NonlinearConstraint(constraint_func, bound, bound)


    # target function for minimization
    def dyson_difference(x):
        """Target function for minimize."""
        _sig_iwaa = Gf(mesh=mesh_iw, data=unflatten(x))
        _sig_iwaa += sigma_moments[0]
        #  G - G0 - G0*Σ*G = 0 done on the DLR nodes
        r_iwaa = g_iwaa - g0_iwaa - g0_iwaa * _sig_iwaa * g_iwaa
        # the Frobeinus norm
        r = np.sqrt(np.sum(r_iwaa.tau_L2_norm() ** 2))
        return r

    # compute initial guess for Sigma from Dyson equation
    sig0_iwaa = inverse(g0_iwaa) - inverse(g_iwaa) - sigma_moments[0]
    x_init = flatten(sig0_iwaa.data)

    # run solver to optimize Σ(iν)
    sol = minimize(dyson_difference, x_init, method=method, constraints=constraints, options=options)  # type: ignore

    if verbosity > 0:
        mpi.report(sol.message)
    if not sol.success:
        mpi.report('[WARNING] Minimization did not converge! Please proceed with caution!')

    # create optimized self-energy from minimizer
    sig_iwaa = Gf(mesh=mesh_iw, data=unflatten(sol.x))

    # mpi.report(f'L2 norm of residual (G-G₀-G₀ΣG): {solution.fun:.4e}')
    if verbosity > 1 and len(sigma_moments) >= 2:
        constraint_violation = np.max(np.abs(make_gf_dlr(sig_iwaa).data.sum(axis=0) - sigma_moments[1]))
        mpi.report(f'Σ1 constraint diff: {constraint_violation:.4e}')

    residual = sol.fun
    if scalar_output:
        return sig_iwaa[0, 0], sigma_moments[0][0, 0], residual
    else:
        return sig_iwaa, sigma_moments[0], residual


def crm_solve_dyson(
    g_tau: Union[Gf, BlockGf],
    g0_iw: Union[Gf, BlockGf],
    sigma_moments: Union[np.ndarray, Dict[str, np.ndarray]],
    w_max: Union[Dict[str, float], float],
    eps: float = 1e-6,
    hermitian: bool = True,
    verbosity: int = 0,
    **kwargs,
) -> Union[Gf, BlockGf]:
    """Solve the Dyson equation via a constrained minimization problem (CRM).

    Parameters
    ----------
    g_tau : imaginary time Gf or BlockGf
        The imaginary time Green's function measured by the impurity solver.
    g0_iw : imaginary frequency gf or BlockGf
        The non-interacting Green's function.
    sigma_moments : np.ndarray or dict of np.ndarray
        The moments of the self-energy to be used in the CRM solver.
    w_max : float or dict of float
        Spectral width of the impurity problem for DLR basis.
    eps : float
        Accuracy of the DLR basis to represent Green’s function. Defaults to 1e-6.
    hermitian : bool, optional
        Whether to enforce hermiticity of the self-energy. Defaults to True.
    verbosity: int, optional
        Verbosity level for mpi reporting (0: warnings, 1: info, 2: full). Defaults to 0 (warnings)
    **kwargs
        Additional keyword arguments for `minimize_dyson`.

    Returns
    -------
    sigma: imaginary frequency Gf or BlockGf
        The self-energy obtained by solving the Dyson equation via CRM.

    References
    ----------
    [1] https://arxiv.org/abs/2310.01266.
    """

    kwargs.update({"verbosity": verbosity})

    # recursive call for BlockGf
    if isinstance(g_tau, BlockGf) or isinstance(g0_iw, BlockGf):
        assert isinstance(g_tau, BlockGf) and isinstance(g0_iw, BlockGf), "g_tau and g0_iw must be both Gf or BlockGf"
        assert list(g_tau.indices).sort() == list(g0_iw.indices).sort(), "g_tau and g0_iw must have the same block structure"
        assert isinstance(sigma_moments, dict), "sigma_moments must be a dict when g_tau and g0_iw are BlockGf"

        # If wmax is a float, use the same for all blocks
        if not isinstance(w_max, dict):
            w_max = {block: w_max for block in g_tau.indices}

        block_list = []

        for block, g in g_tau:
            if verbosity > 0:
                mpi.report(f"Block {block}...")
            sig = crm_solve_dyson(g, g0_iw[block], sigma_moments[block], w_max[block], eps, hermitian, **kwargs)
            block_list.append(sig)

        return BlockGf(name_list=list(g_tau.indices), block_list=block_list)

    assert not isinstance(sigma_moments, dict), "sigma_moments must be a ndarray when g_tau and g0_iw are Gf"
    assert not isinstance(w_max, dict), "w_max must be a float when g_tau and g0_iw are Gf"

    # Fit DLR Green’s function to imaginary time Green’s function
    g_dlr_iw = fit_gf_dlr(g_tau, w_max=w_max, eps=eps)

    # Read off G0 at the DLR nodes
    mesh_iw = MeshDLRImFreq(g_dlr_iw.mesh)
    g0_dlr_iw = Gf(mesh=mesh_iw, target_shape=g_dlr_iw.target_shape)
    for iwn in mesh_iw:
        g0_dlr_iw[iwn] = g0_iw(iwn.value)

    # Use the CRM solver to minimize the Dyson error
    sigma_dlr, s_hf, residual = minimize_dyson(g0_dlr_iw, g_dlr_iw, sigma_moments, **kwargs)

    # Since a spectral representable G has no constant we have to manually add the Hartree
    # shift after the solver is finished again
    sigma_iw = make_gf_imfreq(sigma_dlr, n_iw=g0_iw.mesh.n_iw)
    sigma_iw += s_hf

    return make_hermitian(sigma_iw) if hermitian else sigma_iw
