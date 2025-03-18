# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-07-03

import string
from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from triqs.gf import BlockGf, Gf, inverse

from .convergence import max_difference
from .functions import Ht
from .utility import GfLike, apply_mixing, blockgf, report, toarray

__all__ = [
    "generate_cmpt_names",
    "initialize_gf_cmpt",
    "initalize_onsite_energy",
    "gf_coherent",
    "gf_component",
    "solve_iter",
    "solve_cpa",
]

Onsite = Union[Sequence[Union[float, Sequence[float]]], BlockGf]


def generate_cmpt_names(n_cmpt: int) -> List[str]:
    return list(string.ascii_uppercase[:n_cmpt])


def initialize_gf_cmpt(
    sigma_cpa: GfLike, cmpts: Union[int, Sequence[str]], name: str = "G_cmpt"
) -> BlockGf:
    if isinstance(cmpts, int):
        cmpts = generate_cmpt_names(cmpts)
    return blockgf(sigma_cpa.mesh, names=list(cmpts), blocks=sigma_cpa, name=name, copy=True)


def initalize_onsite_energy(
    sigma_cpa: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    cmpt_names: Sequence[str] = None,
    name: str = "eps",
) -> BlockGf:
    if isinstance(eps, BlockGf):
        return eps

    # Remove values with c=0
    conc, eps = zip(*[(c, e) for c, e in zip(conc, eps) if c > 0])
    # Broadcast shapes
    is_block = isinstance(sigma_cpa, BlockGf)
    n_blocks = len(sigma_cpa) if is_block else 0
    n_cmpt = len(conc)
    eps = np.asarray(eps)
    if is_block and len(eps.shape) == 1:
        eps = np.array([eps.copy() for _ in range(n_blocks)]).swapaxes(0, 1)

    names = generate_cmpt_names(n_cmpt) if cmpt_names is None else list(cmpt_names)
    blocks = [sigma_cpa] * n_cmpt
    eps_eff = blockgf(sigma_cpa.mesh, names=names, blocks=blocks, name=name, copy=True)
    for i, (name, eps_i) in enumerate(eps_eff):
        if is_block:
            ei = [eps[i]] * n_cmpt if not hasattr(eps[i], "__len__") else eps[i]
            for s, (spin, eps_is) in enumerate(eps_i):
                eps_is.data[:] = ei[s]
        else:
            eps_i << eps[i]

    return eps_eff


def _validate(
    sigma: GfLike, conc: Sequence[float], eps: Onsite
) -> Tuple[bool, np.ndarray, Union[BlockGf, np.ndarray]]:
    if isinstance(eps, BlockGf):
        # Convert back to numpy array if eps is a Block Gf
        # ToDo: Implement CPA methods to accept BlockGf
        eps = toarray(eps)

    # Remove values with c=0
    conc, eps = zip(*[(c, e) for c, e in zip(conc, eps) if c > 0])
    # Broadcast shapes
    is_block = isinstance(sigma, BlockGf)
    n_blocks = len(sigma) if is_block else 0
    n_cmpt = len(conc)
    conc = np.asarray(conc)
    eps = np.asarray(eps)
    if is_block and len(eps.shape) == 1:
        eps = np.array([eps.copy() for _ in range(n_blocks)]).swapaxes(0, 1)
    # Check if arguments are valid
    if sum(conc) != 1.0:
        raise ValueError(f"Sum of concentrations {list(conc)} does not add up to 1!")
    if eps.shape[0] != n_cmpt:
        raise ValueError(f"Shape mismatch of eps {eps.shape} and number of components {n_cmpt}!")
    if is_block and eps.shape[1] != n_blocks:
        raise ValueError(f"Shape mismatch of eps {eps.shape} and number of blocks {n_blocks}!")
    return is_block, conc, eps


def gf_coherent(ht: Ht, sigma: GfLike, eta: float = 0.0, name: str = "G_coh") -> GfLike:
    """Compute the coherent (total) Green's functions `G_h(z)`.

    Parameters
    ----------
    ht : Ht
        Lattice Hilbert transformation used to calculate the coherent Green's function.
    sigma : Gf or BlockGf
        The SSA self-energy as TRIQS gf.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object.

    Returns
    -------
    Gf or BlockGf
        The coherent Green's function embedded in `sigma`.
    """
    gc = sigma.copy()
    gc.name = name

    if isinstance(sigma, Gf):
        gc << ht(sigma, eta=eta)

    elif isinstance(sigma, BlockGf):
        for name, g in gc:
            g << ht(sigma[name], eta=eta)

    else:
        raise ValueError(f"Invalid type of `sigma`: {type(sigma)}.")
    return gc


def gf_component(
    ht: Ht,
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    eta: float = 0.0,
    scale: bool = False,
    cmpt_names: list = None,
    name: str = "G_cmpt",
) -> BlockGf:
    """Compute the component Green's functions `G_i(z)`.

    .. math::
        G_i(z) = c_i G_i(z) / (1 - (ε_i - Σ(z)) G_0(z - Σ(z)) )

    Parameters
    ----------
    ht : Ht
        Lattice Hilbert transformation used to calculate the coherent Green's function.
    sigma : Gf or BlockGf
        The SSA self-energy as TRIQS gf.
    conc : (..., N_cmpt) float array_like, optional
        Concentration of the different components used for the average.
        If not provided, the component GFs are returned unweighted.
    eps : (N_cmpt, [N_spin], ...) array_like
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    scale : bool, optional
        If True, the component Gfs are multiplied by the concentrations. In this case
        the sum of the component Gfs is equal to the coherent Green's function.
        The default is `False`.
    cmpt_names : list, optional
        List of names for the components. If not provided, the names are upper case
        ASCII characters `A`, `B`, `C`, ...
    name : str, optional
        The name of the resulting Gf object.

    Returns
    -------
    BlockGf
        The Green's function of the components embedded in `sigma_cpa`.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    n_cmpt = len(conc)
    if cmpt_names is None:
        cmpt_names = list(eps.indices) if isinstance(eps, BlockGf) else generate_cmpt_names(n_cmpt)

    g_cmpt = blockgf(sigma.mesh, cmpt_names, blocks=[sigma] * n_cmpt, name=name, copy=True)
    cc = conc if scale else np.ones_like(conc)

    def _g_cmpt(_sig: Gf, _eps_i: ArrayLike, _conc_i: ArrayLike) -> Gf:
        """Compute the component Green's functions for a single Gf."""
        _g = _sig.copy()
        _g << ht(_sig, eta=eta)
        _g.data[:] = _conc_i * _g.data / (1 - _g.data * (_eps_i - _sig.data))
        return _g

    if isinstance(sigma, Gf):
        for i, (name, g_i) in enumerate(g_cmpt):
            g_i << _g_cmpt(sigma, eps[i], cc[i])

    elif isinstance(sigma, BlockGf):
        for i, (name, g_i) in enumerate(g_cmpt):
            for s, (spin, g_s) in enumerate(g_i):
                g_s << _g_cmpt(sigma[spin], eps[i, s], cc[i])

    else:
        raise ValueError(f"Invalid type of `sigma`: {type(sigma)}.")

    return g_cmpt


# -- VCA Solve methods -----------------------------------------------------------------------------


def solve_vca(
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    name: str = "Σ_vca",
) -> GfLike:
    """Solve the CPA equations using the VCA approximation.

    The virtual crystal approximation (VCA) is the simplest form of the CPA.
    The self-energy is given by the average of the site self-energies weighted by the concentration.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)
    if is_block:
        for i, name in enumerate(sigma.indices):
            sigma[name].data[:] = np.sum(eps[:, i] * conc)
    else:
        sigma.data[:] = np.sum(eps * conc)

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


# -- ATA solve methods -----------------------------------------------------------------------------


def solve_ata(
    ht: Ht,
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    eta: float = 0.0,
    name: str = "Σ_ata",
) -> GfLike:
    """Solve the CPA equations using the ATA approximation."""
    is_block, conc, eps = _validate(sigma, conc, eps)

    sigma_vca = sigma.copy()
    sigma_vca.zero()

    g0 = sigma.copy()
    g0.zero()

    # Unperturbated Green's function (uses VCA)
    if is_block:
        for i, name in enumerate(sigma_vca.indices):
            sigma_vca[name].data[:] = np.sum(eps[:, i] * conc)
            g0[name] << ht(sigma_vca[name], eta=eta)
    else:
        sigma_vca.data[:] = np.sum(eps * conc)
        g0 << ht(sigma_vca, eta=eta)

    # Average T-matrix
    def _tavrg(_g0: Gf, _eps: np.ndarray, _sigma: Gf) -> Gf:
        _tmat = _g0.copy()
        _cmpts = [(_e - _sigma.data) / (1 - (_e - _sigma.data) * _g0.data) for _e in _eps]
        _tmat.data[:] = np.sum([_c * _g.data for _c, _g in zip(conc, _cmpts)], axis=0)
        return _tmat

    tavrg = sigma.copy()
    if is_block:
        for s, (name, t) in enumerate(tavrg):
            t << _tavrg(g0[name], eps[:, s], sigma_vca[name])
    else:
        tavrg << _tavrg(g0, eps, sigma_vca)

    sigma << tavrg * inverse(1 + g0 * tavrg) + sigma_vca

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


# -- CPA solve methods -----------------------------------------------------------------------------


def solve_iter(
    ht: Ht,
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    eta: float = 0.0,
    name: str = "Σ_cpa",
    tol: float = 1e-6,
    mixing: float = 1.0,
    maxiter: int = 1000,
    verbosity: int = 1,
) -> GfLike:
    """Determine the CPA self-energy by an iterative solution of the CPA equations.

    Parameters
    ----------
    ht : Ht
        Lattice Hilbert transformation used to calculate the coherent Green's function.
    sigma : Gf or BlockGf
        Starting guess for CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) array_like or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.
    tol : float, optional
        The tolerance for the convergence of the CPA self-energy.
        The iteration stops when the norm between the old and new self-energy
        .math:`|Σ_new - Σ_old|` is smaller than `tol`.
    mixing : float, optional
        The mixing parameter for the self-energy update. The new self-energy is
        computed as `Σ_new = (1 - mixing) * Σ_old + mixing * Σ_new`.
    maxiter : int, optional
        The maximum number of iterations, by default 1000.
    verbosity : {0, 1, 2} int, optional
        The verboisity level.

    Returns
    -------
     Gf or BlockGf
        The self-consistent CPA self energy `Σ_c`. Same as thew input self energy after
        calling the method.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    # Skip trivial solution
    if len(conc) == 1:
        report("Single component, skipping CPA!")
        if is_block:
            for i, name in enumerate(sigma.indices):
                sigma[name].data[:] = eps[0, i]
        else:
            sigma.data[:] = eps[0]
        return sigma

    report(f"Solving CPA problem for {len(conc)} components iteratively...")
    # Initial coherent Green's function
    gc = sigma.copy()
    gc.name = "Gc"
    gc.zero()

    # Old self energy for convergence check
    sigma_old = sigma.copy()

    # Define avrg G method
    def _g_avrg(_g0_inv: Gf, _eps: np.ndarray) -> Gf:
        """Compute component and average Green's function `<G>(z) = ∑ᵢ cᵢ Gᵢ(z)`."""
        _g_i = _g0_inv.copy()
        _cmpts = [1 / (_g0_inv.data - _e) for _e in _eps]
        _g_i.data[:] = np.sum([_g.data * _c for _c, _g in zip(conc, _cmpts)], axis=0)
        return _g_i

    # Begin CPA iterations
    for it in range(maxiter):
        # Compute average GF via the self-energy:
        # <G> = G_0(E - Σ) = 1 / (E - H_0 - Σ)
        if is_block:
            for s, (spin, g) in enumerate(gc):
                g << ht(sigma[spin], eta=eta)
        else:
            gc << ht(sigma, eta=eta)

        # Compute non-interacting GF via Dyson equation
        g0_inv = sigma + inverse(gc)

        # g_i = [inverse(sigma - eps[i] + inverse(gc)) for i, c in enumerate(conc)]
        # Compute new coherent GF: <G> = c_A G_A + c_B G_B + ...
        if is_block:
            for s, (spin, g) in enumerate(gc):
                g << _g_avrg(g0_inv[spin], eps[:, s])
        else:
            gc << _g_avrg(g0_inv, eps)

        # Update self energy via Dyson: Σ = G_0^{-1} - <G>^{-1}
        sigma << g0_inv - inverse(gc)

        # Apply mixing
        apply_mixing(sigma_old, sigma, mixing)

        # Check for convergence
        diff = max_difference(sigma_old, sigma, norm_temp=True, relative=False)
        if verbosity > 1:
            report(f"CPA iteration {it + 1}: Error={diff:.10f}")

        if diff <= tol:
            if verbosity > 0:
                report(f"CPA converged in {it + 1} iterations (Error: {diff:.10f})")
            break
        sigma_old = sigma.copy()
    else:
        if verbosity > 0:
            report(f"CPA did not converge after {maxiter} iterations")

    report("")

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


def solve_cpa(
    ht: Ht,
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    eta: float = 0.0,
    name: str = "Σ_cpa",
    method: str = "iter",
    **kwds,
) -> GfLike:
    """Determine the CPA self-energy of the CPA equations.

    Parameters
    ----------
    ht : Ht
        Lattice Hilbert transformation used to calculate the coherent Green's function.
    sigma : Gf or BlockGf
        Starting guess for CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) complex np.ndarray or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.
    method : {"iter",} str, optional
        The method to use for solving the CPA root equation. Can be either 'iter' for
        the iterative algorythm or 'root' for the optimization algorythm.
    **kwds
        Additional keyword arguments passed to the specif solve method.

    Returns
    -------
     Gf or BlockGf
        The self-consistent CPA self energy `Σ_c`. Same as thew input self energy after
        calling the method.
    """
    supported = {"iter": solve_iter}

    kwds.update(ht=ht, sigma=sigma, eps=eps, conc=conc, eta=eta, name=name)
    try:
        func = supported[method.lower()]
        return func(**kwds)
    except KeyError:
        raise ValueError(f"Invalid method: {method}. Use {list(supported.keys())}!")
