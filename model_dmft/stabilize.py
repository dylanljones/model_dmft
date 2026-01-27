# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2026-01-25

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from triqs.gf import (
    BlockGf,
    Gf,
    MeshDLRImFreq,
    MeshImFreq,
    MeshLegendre,
    fit_gf_dlr,
    make_gf_imfreq,
    make_hermitian,
)

# from triqs.gf.dlr_crm_dyson_solver import minimize_dyson
from triqs.gf.tools import fit_legendre as _fit_legendre
from triqs.utility import mpi

from .crm_solver import minimize_dyson
from .utility import GfLike, dyson, mpi_collect_to_list, mpi_enumerate_array, report, toarray

EPS = 1e-100  # small number to avoid div by zero


def zero_matsubara_index(gf: GfLike) -> int:
    """Returns the index of the first positive Matsubara frequency."""
    n_iw = gf.mesh.n_iw
    mesh_iw = toarray(gf.mesh)
    i0 = n_iw
    assert mesh_iw[i0 - 1] < 0.0
    assert mesh_iw[i0] > 0.0
    return i0


def distance(a: np.ndarray, b: np.ndarray, relative: bool = True, q: float = None) -> float:
    """Scalar metric between two arrays."""
    num = np.abs(a - b)
    den = np.maximum(np.abs(b), EPS)
    err = num / den if relative else num
    err = np.quantile(err, q, axis=0) if q is not None else err
    return float(err if err.ndim == 0 else np.mean(err))


def gf_distance(gf1: GfLike, gf2: GfLike, n_stop: int = None, relative: bool = True, q: float = None) -> float:
    """Scalar metric between two Greens function methods over low Matsubara frequencies."""
    i0 = zero_matsubara_index(gf1)
    a = toarray(gf1)[..., i0:, 0, 0]
    b = toarray(gf2)[..., i0:, 0, 0]
    if a.ndim > 1:
        a = np.swapaxes(a, 0, 1)
        b = np.swapaxes(b, 0, 1)
    if n_stop is not None:
        a = a[:n_stop]
        b = b[:n_stop]
    return distance(a, b, relative=relative, q=q)


def rolling_windows(x: np.ndarray, k: int) -> np.ndarray:
    if k % 2 == 0:
        k += 1
    pad = k // 2
    if x.ndim == 1:
        x_pad = np.pad(x, (pad, pad), mode="edge")
    else:
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    x_win = sliding_window_view(x_pad, k, axis=0)
    assert x_win.shape[0] == x.shape[0]
    return x_win


def find_sigma_ref_cutoff(sigma: GfLike, smooth: int = 5, smooth_err: int = 30, dist: int = 10) -> int:
    i0 = zero_matsubara_index(sigma)
    sig = toarray(sigma)[..., i0:, 0, 0].imag
    if sig.ndim > 1:
        sig = np.swapaxes(sig, 0, 1)  # (N_freq, N_spin)

    sig_win = rolling_windows(sig, smooth)
    sig_errs = 1.4826 * np.median(np.abs(sig_win - np.median(sig_win, axis=-1, keepdims=True)), axis=-1)

    sig_errs_avg = np.mean(rolling_windows(sig_errs, smooth_err), axis=-1)
    if sig_errs_avg.ndim > 1:
        sig_errs_avg = np.mean(sig_errs_avg, axis=-1)

    peaks, _ = find_peaks(-sig_errs_avg, distance=dist)
    return peaks[0] if len(peaks) > 0 else None


def find_first_consecutive(ok: np.ndarray, consec: int = 1) -> int:
    run = 0
    chosen = None
    for i in range(len(ok)):
        run = run + 1 if ok[i] else 0
        if run >= consec:
            chosen = int(i - consec + 1)
            break
    return chosen


# ---- Tail fitting --------------------------------------------------------------------------------


class TailPolynomial:
    def __init__(self, coeffs: dict[str, ArrayLike]):
        data = np.array([coeffs[k] for k in coeffs.keys()])
        self.coeffs = data.swapaxes(0, 1)  # (N_moments, N_spin)

    def __call__(self, x: Union[complex, ArrayLike, MeshImFreq]) -> np.ndarray:
        if isinstance(x, MeshImFreq):
            x = toarray(x)
        x = np.array(x, dtype=complex)
        res = np.zeros((*x.shape, *self.coeffs.shape[1:]), dtype=complex)
        for n, c in enumerate(self.coeffs):
            coeff = c[np.newaxis, ...]
            power = np.power(x, n)[..., np.newaxis, np.newaxis, np.newaxis]
            res += coeff / power
        res = np.swapaxes(res, 0, 1)
        return res


# ---- CRM methods ---------------------------------------------------------------------------------


def crm_solve_dyson(
    g_tau: BlockGf, g0_iw: BlockGf, sigma_moments: dict[str, np.ndarray], w_max: float, eps: float, verbosity: int = 0
) -> BlockGf:
    """Solve the Dyson equation via a constrained minimization problem (CRM).

    Parameters
    ----------
    g_tau : imaginary time BlockGf
        The imaginary time Green's function measured by the impurity solver.
    g0_iw : imaginary frequency BlockGf
        The non-interacting Green's function.
    sigma_moments : dict[str, np.ndarray]
        The moments of the self-energy to be used in the CRM solver.
    w_max : float
        Spectral width of the impurity problem for DLR basis.
    eps : float
        Accuracy of the DLR basis to represent Green’s function

    Returns
    -------
    sigma: imaginary frequency BlockGf
        The self-energy obtained by solving the Dyson equation via CRM.

    References
    ----------
    [1] https://arxiv.org/abs/2310.01266.
    """
    # Fit DLR Green’s function to imaginary time Green’s function
    g_dlr_iw = fit_gf_dlr(g_tau, w_max=w_max, eps=eps)

    # Read off G0 at the DLR nodes
    names = list(g_tau.indices)
    mesh_iw = MeshDLRImFreq(g_dlr_iw.mesh)
    g = Gf(mesh=mesh_iw, target_shape=g_dlr_iw[names[0]].target_shape)
    g0_dlr_iw = BlockGf(name_list=names, block_list=[g, g])
    for name, g in g0_dlr_iw:
        for iwn in mesh_iw:
            g[iwn] = g0_iw[name](iwn.value)

    g = Gf(mesh=mesh_iw, target_shape=g_dlr_iw[names[0]].target_shape)
    sigma_dlr = BlockGf(name_list=names, block_list=[g, g])
    n_iw = g0_iw.mesh.n_iw

    # Use the CRM solver to minimize the Dyson error
    for name, sig in sigma_dlr:
        s_dlr, s_hf, residual = minimize_dyson(
            G0_dlr=g0_dlr_iw[name], G_dlr=g_dlr_iw[name], Sigma_moments=sigma_moments[name], verbosity=verbosity
        )
        sig << s_dlr  # noqa

    # Since a spectral representable G has no constant we have to manually add the Hartree
    # shift after the solver is finished again
    sigma_iw = make_gf_imfreq(sigma_dlr, n_iw=n_iw)
    for name, sig in sigma_iw:
        sig += sigma_moments[name][0]

    return make_hermitian(sigma_iw)


@dataclass
class WMaxOptResult:
    success: bool
    index: Optional[int]
    wmax: Optional[float]
    sigma_opt: Optional[BlockGf]
    wmax_grid: np.ndarray
    metrics: np.ndarray
    metrics_ref: np.ndarray
    metrics_tails: np.ndarray
    sigmas: list[BlockGf]
    tol: float


def pick_wmax_opt(
    g_tau: BlockGf,
    g0_iw: BlockGf,
    g_iw: BlockGf,
    sigma_moments: dict[str, np.ndarray],
    *,
    start: float = 1.0,
    stop: float = 5.0,
    step: float = 0.1,
    smooth: int = 8,
    smooth_err: int = 30,
    iw_noise: int = None,
    iw_stop: int = 50,
    tail_frac: float = 0.8,
    idx_step: int = 1,
    tol: float = 0.01,
    q: float = None,
    consec: int = 2,
) -> WMaxOptResult:
    wmax_grid = np.round(np.arange(start, stop, step), decimals=3)
    n = len(wmax_grid)

    sigma_ref = dyson(g0=g0_iw, g=g_iw)
    if iw_noise is None:
        cutoff = find_sigma_ref_cutoff(sigma=sigma_ref, smooth=smooth, smooth_err=smooth_err, dist=10)
        cutoff = int(0.5 * cutoff) if cutoff is not None else 10
    else:
        cutoff = iw_noise

    # compute Sigma(wmax) for each wmax
    # sigmas = list()
    sigma_pairs = list()
    for i, wmax in mpi_enumerate_array(wmax_grid):
        report(f"Solving CRM for wmax: {wmax}", once=False, rank=True)
        sigma_crm = crm_solve_dyson(g_tau, g0_iw, sigma_moments, w_max=wmax, eps=1e-6)
        # report("")
        # sigmas.append(sigma_crm)
        sigma_pairs.append((i, sigma_crm))
    mpi.barrier()
    sigmas: List[GfLike] = mpi_collect_to_list(sigma_pairs, length=n, root=0)  # type: ignore
    mpi.barrier()

    if mpi.is_master_node():
        assert len(sigmas) == n, "Sigma list has incorrect length."
        assert all(s is not None for s in sigmas), "Some Sigma computations failed."
        # build metric curve m(i)
        metrics = []
        pairs = list(zip(range(n - idx_step), range(idx_step, n)))
        for i, j in pairs:
            dist = gf_distance(sigmas[j], sigmas[i], n_stop=iw_stop, relative=True, q=q)
            metrics.append(dist)
        metrics = np.asarray(metrics, float)
        metrics = np.pad(metrics, (idx_step, 0), mode="edge")  # align sizes

        # build metric curve m(i) to reference
        metrics_ref = []
        for i in range(n):
            dist = gf_distance(sigmas[i], sigma_ref, n_stop=cutoff, relative=True, q=q)
            metrics_ref.append(dist)
        metrics_ref = np.asarray(metrics_ref, float)

        metrics_tails = list()
        tail_poly = TailPolynomial(sigma_moments)
        n_iw = g0_iw.mesh.n_iw
        i0 = int(n_iw * (1.0 + tail_frac))  # last % of Matsubara points
        iw = toarray(g0_iw.mesh)[i0:]
        tails = tail_poly(1j * iw)[..., 0, 0].imag
        for sigma in sigmas:
            data = toarray(sigma)[..., i0:, 0, 0].imag
            dist = distance(tails, data, relative=True, q=q)
            metrics_tails.append(dist)
        metrics_tails = np.asarray(metrics_tails, float)

        ok = np.logical_and(metrics <= tol, metrics_ref <= tol)
        ok = np.logical_and(ok, metrics_tails <= tol)
        chosen = find_first_consecutive(ok, consec=consec)
        success = chosen is not None
        res = WMaxOptResult(
            success=success,
            index=chosen,
            wmax=wmax_grid[chosen] if chosen is not None else None,
            sigma_opt=sigmas[chosen] if chosen is not None else None,
            wmax_grid=wmax_grid,
            metrics=metrics,
            metrics_ref=metrics_ref,
            metrics_tails=metrics_tails,
            sigmas=sigmas,
            tol=tol,
        )
    else:
        res = None
    res = mpi.bcast(res)
    return res


def plot_wmax_metrics(res: WMaxOptResult) -> None:
    n = len(res.metrics)
    plt.plot(res.wmax_grid[:n], res.metrics, label="Σ conv")
    plt.plot(res.wmax_grid[:n], res.metrics_ref, label="Σ ref")
    plt.plot(res.wmax_grid[:n], res.metrics_tails, label="Σ tail")
    plt.axhline(res.tol, color="k", ls="--")
    if res.success:
        plt.axvline(res.wmax, color="r", ls="--")
    plt.xlabel("$w_{max}$")
    plt.ylabel("Error")
    plt.legend()


# ---- Legendre methods ----------------------------------------------------------------------------


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


def truncate_g_l(g_l_orig: Union[BlockGf, Gf], n_l: int) -> Union[BlockGf, Gf]:
    """Truncate a Legendre Green's function to a specified number of Legendre coefficients.

    Parameters
    ----------
    g_l_orig : BlockGf
        The original Legendre Green's function to be truncated.
    n_l : int
        The number of Legendre coefficients to retain.

    Returns
    -------
    BlockGf
        A new Legendre Green's function truncated to the specified number of coefficients.
    """
    # Extract the mesh from the original Green's function
    mesh_orig = g_l_orig.mesh
    # Create a new Legendre mesh with the specified maximum number of coefficients
    mesh = MeshLegendre(beta=mesh_orig.beta, statistic=mesh_orig.statistic, max_n=n_l)

    def _truncate(_g: Gf) -> Gf:
        return Gf(mesh=mesh, data=_g.data[:n_l])

    if isinstance(g_l_orig, BlockGf):
        names, blocks = list(), list()
        # Iterate over the blocks in the original Green's function
        for name, g in g_l_orig:
            names.append(name)
            blocks.append(_truncate(g))
        return BlockGf(name_list=names, block_list=blocks, name=g_l_orig.name)
    else:
        return _truncate(g_l_orig)


def from_legendre(g0: GfLike, g_l: GfLike) -> Union[Gf, BlockGf]:
    """Compute a Gf object from Legendre coefficients."""
    g_out = g0.copy()

    def _set_from_legendre(_g: Gf, _g_l: Gf) -> None:
        _g_l.enforce_discontinuity(np.identity(_g_l.target_shape[0]))  # noqa
        _g.set_from_legendre(_g_l)  # noqa

    if isinstance(g0, BlockGf):
        for name, g in g_out:
            _set_from_legendre(g, g_l[name])
    else:
        _set_from_legendre(g_out, g_l)

    return g_out


def sigma_from_legendre(g0_iw: GfLike, g_l: GfLike) -> Union[Gf, BlockGf]:
    """Compute Sigma(iw) from Legendre coefficients."""
    g_iw = from_legendre(g0_iw, g_l)
    return dyson(g0=g0_iw, g=g_iw)


def legendre_fit(g0_iw: BlockGf, g_iw: BlockGf, g_tau: BlockGf, g_l: BlockGf) -> Tuple[BlockGf, BlockGf, BlockGf]:
    """Fit the Green's functions and self energy using the Legendre Green's function."""
    g_iw_l = g_iw.copy()
    g_tau_l = g_tau.copy()
    for name, g in g_l:
        g.enforce_discontinuity(np.identity(g.target_shape[0]))
        g_iw_l[name].set_from_legendre(g)
        g_tau_l[name].set_from_legendre(g)

    g_iw_l << make_hermitian(g_iw_l)
    sigma_iw_l = dyson(g0=g0_iw, g=g_iw_l)
    return g_iw_l, g_tau_l, sigma_iw_l


def changepoint_bic(g_l: GfLike, min_head: int = 10, min_tail: int = 20, smooth: int = 3) -> Optional[int]:
    """Returns k (inclusive) where decay region ends and noise floor begins at k+1.

    Parameters
    ----------
    g_l : (L, ) np.ndarray
        The Legendre coefficients.
    min_head : int, optional
        Minimum number of coefficients in the head segment. By default 10.
    min_tail : int, optional
        Minimum number of coefficients in the tail segment. By default 10.
    smooth : int, optional
        Smoothing window size (moving average). By default 1 (no smoothing).

    Returns
    -------
    int
        The BIC value for the optimal change-point, or `None` if not found.
    """
    g_l = toarray(g_l)[..., 0, 0]
    if g_l.ndim > 1:
        g_l = np.swapaxes(g_l, 0, 1)  # (N_l, N_spin)
        has_spin = True
    else:
        has_spin = False

    amp = np.abs(g_l)
    nl = len(amp)
    ll = np.arange(nl, dtype=float)

    # optional median smoothing on log amplitude
    eps = np.max(amp) * EPS + EPS
    y = np.log(np.maximum(amp, eps))

    if smooth > 1:
        gl_win = rolling_windows(g_l, smooth)
        y = np.median(gl_win, axis=-1)
    best = None  # (bic, k)
    n = nl
    p = 3  # line: a,b plus tail constant: c

    for k in range(min_head, nl - min_tail):
        # head fit: y = a + b*l
        xh = ll[: k + 1]
        yh = y[: k + 1]
        astack = np.vstack([np.ones_like(xh), xh]).T
        a, b = np.linalg.lstsq(astack, yh, rcond=None)[0]
        if has_spin:
            a = a[:, np.newaxis]
            b = b[:, np.newaxis]
        yhat_h = a + b * xh
        sse_h = np.sum((yh - yhat_h.T) ** 2, axis=0)

        # tail fit: constant
        yt = y[k + 1 :]
        c = np.mean(yt)
        sse_t = np.sum((yt - c) ** 2)

        sse = np.mean(sse_h + sse_t)
        if sse <= 0:
            continue

        bic = n * np.log(sse / n) + p * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k)

    return None if best is None else int(best[1])


def mad_sigma(x: ArrayLike) -> float:
    """Robust sigma estimate from MAD (real-valued)."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def mad_noise_power(z: ArrayLike) -> float:
    """Estimate E[|noise|^2] from complex samples Z using MAD on real and imag.

    Works with arbitrary shape; flattens.
    """
    z = np.asarray(z)
    re = z.real.ravel()
    im = z.imag.ravel()
    return mad_sigma(re) ** 2 + mad_sigma(im) ** 2  # approx E[|noise|^2]


def wiener_shrinkage(g_l: GfLike, noise_start: int = None, noise_power: int = None) -> Tuple[GfLike, np.ndarray, float]:
    mesh = g_l.mesh
    names = [name for name, _ in g_l] if isinstance(g_l, BlockGf) else None
    g_l = toarray(g_l)[..., 0, 0]
    if g_l.ndim > 1:
        g_l = np.swapaxes(g_l, 0, 1)  # (N_l, N_spin)
        has_spin = True
    else:
        has_spin = False

    amp = np.abs(g_l)  # (L,)
    nl = len(amp)

    if noise_power is None:
        if noise_start is None or noise_start >= nl:
            raise ValueError("Provide noise_start within [0, L) or provide noise_power explicitly.")
        noise_power = mad_noise_power(g_l[noise_start:])

    # per-l Wiener weight based on Frobenius amplitude
    sig2 = np.maximum(amp**2 - noise_power, 0.0)
    w_l = sig2 / (sig2 + noise_power + EPS)

    # apply per-l weight
    g_f = w_l * g_l
    if has_spin:
        data = g_f.swapaxes(0, 1)[..., np.newaxis, np.newaxis]  # (N_spin, N_l, 1)
        blocks = [Gf(mesh=mesh, data=data[i], name=name) for i, name in enumerate(names)]
        g_out = BlockGf(name_list=names, block_list=blocks, name="G_l_filt")
    else:
        g_out = Gf(mesh=mesh, data=g_f.reshape((-1, 1, 1)), name="G_l_filt")
    return g_out, w_l, float(noise_power)


@dataclass
class NlOptResult:
    success: bool
    index: Optional[int]
    nl: Optional[int]
    sigma_opt: Optional[BlockGf]
    nl_grid: np.ndarray
    metrics: np.ndarray
    metrics_ref: np.ndarray
    metrics_tails: np.ndarray
    sigmas: list[BlockGf]
    tol: float


def pick_nl_opt(
    g_l: BlockGf,
    g0_iw: BlockGf,
    g_iw: BlockGf,
    sigma_moments: Union[dict[str, np.ndarray], None],
    *,
    nl_step: int = 2,
    smooth: int = 8,
    smooth_err: int = 30,
    iw_noise: int = None,
    iw_stop: int = 50,
    tail_frac: float = 0.8,
    idx_step: int = 1,
    tol: float = 0.01,
    q: float = None,
    consec: int = 1,
) -> NlOptResult:
    nl_max = len(g_l.mesh)
    nl_grid = np.arange(nl_step, nl_max + 1, nl_step, dtype=int)
    n = len(nl_grid)

    sigma_ref = dyson(g0=g0_iw, g=g_iw)
    if iw_noise is None:
        cutoff = find_sigma_ref_cutoff(sigma=sigma_ref, smooth=smooth, smooth_err=smooth_err, dist=10)
        cutoff = int(0.5 * cutoff) if cutoff is not None else 10
    else:
        cutoff = iw_noise

    # compute Sigma(nl) for each nl
    sigmas = list()
    for i, nl in enumerate(nl_grid):
        gl_trunc = truncate_g_l(g_l, nl)
        sigma = sigma_from_legendre(g0_iw, gl_trunc)
        sigmas.append(sigma)

    # build metric curve m(i)
    metrics = []
    pairs = list(zip(range(n - idx_step), range(idx_step, n)))
    for i, j in pairs:
        dist = gf_distance(sigmas[j], sigmas[i], n_stop=iw_stop, relative=True, q=q)
        metrics.append(dist)
    metrics = np.asarray(metrics, float)
    metrics = np.pad(metrics, (idx_step, 0), mode="edge")  # align sizes

    # build metric curve m(i) to reference
    metrics_ref = []
    for i in range(n):
        dist = gf_distance(sigmas[i], sigma_ref, n_stop=cutoff, relative=True, q=q)
        metrics_ref.append(dist)
    metrics_ref = np.asarray(metrics_ref, float)

    metrics_tails = list()
    if sigma_moments is not None:
        tail_poly = TailPolynomial(sigma_moments)
        n_iw = g0_iw.mesh.n_iw
        i0 = int(n_iw * (1.0 + tail_frac))  # last % of Matsubara points
        iw = toarray(g0_iw.mesh)[i0:]
        tails = tail_poly(1j * iw)[..., 0, 0].imag
        for sigma in sigmas:
            data = toarray(sigma)[..., i0:, 0, 0].imag
            dist = distance(tails, data, relative=True, q=q)
            metrics_tails.append(dist)
        metrics_tails = np.asarray(metrics_tails, float)
    else:
        metrics_tails = np.zeros_like(metrics)

    ok = np.logical_and(metrics <= tol, metrics_ref <= tol)
    ok = np.logical_and(ok, metrics_tails <= tol)
    chosen = find_first_consecutive(ok, consec=consec)
    success = chosen is not None
    return NlOptResult(
        success=success,
        index=chosen,
        nl=nl_grid[chosen] if chosen is not None else None,
        sigma_opt=sigmas[chosen] if chosen is not None else None,
        nl_grid=nl_grid,
        metrics=metrics,
        metrics_ref=metrics_ref,
        metrics_tails=metrics_tails,
        sigmas=sigmas,
        tol=tol,
    )


def plot_nl_metrics(res: NlOptResult) -> None:
    n = len(res.metrics)
    plt.plot(res.nl_grid[:n], res.metrics, label="Σ conv")
    plt.plot(res.nl_grid[:n], res.metrics_ref, label="Σ ref")
    plt.plot(res.nl_grid[:n], res.metrics_tails, label="Σ tail")
    plt.axhline(res.tol, color="k", ls="--")
    if res.success:
        plt.axvline(res.nl, color="r", ls="--")
    plt.xlabel("$N_{l}$")
    plt.ylabel("Error")
    plt.legend()
