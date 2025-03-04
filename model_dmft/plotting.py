# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-14

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

from .utility import DN, UP


def crosshair(
    ax: plt.Axes,
    show_x: bool = True,
    show_y: bool = True,
    x: float = 0,
    y: float = 0,
    color: str = "dimgrey",
    lw: float = 0.8,
    axis_below: bool = True,
) -> None:
    ax.set_axisbelow(axis_below)
    if show_y:
        ax.axhline(y, color=color, lw=lw)
    if show_x:
        ax.axvline(x, color=color, lw=lw)


def plot_errors(
    output_file: str, u: float = None, de: float = None, log: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    iterations = list()
    errors_gf = list()
    errors_sigma = list()
    with HDFArchive(str(output_file), "r") as ar:
        max_it = ar["it"]  # Get latest iteration number
        for i in range(1, max_it + 1):
            iterations.append(i)
            err_g = ar[f"err_g-{i}"]
            err_s = ar[f"err_sigma-{i}"]
            errors_gf.append(err_g)
            errors_sigma.append(err_s)

    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(iterations, errors_gf, "-", label="Gf coh")
    ax.plot(iterations, errors_sigma, "-", label="Sigma")
    ax.set(xlabel="Iteration", ylabel="Error", xlim=(0, None))
    ax.legend()
    ax.grid()
    if log:
        ax.set_yscale("log")
    fig.tight_layout()

    return fig, ax


def plot_greens_functions(
    output_file: str, it: int = -1, u: float = None, de: float = None
) -> Tuple[plt.Figure, plt.Axes]:
    with HDFArchive(output_file, "r") as ar:
        if it < 0:
            try:
                it = ar["it"]  # Get latest iteration number
            except KeyError:
                raise ValueError("No iteration completed.")

        g_coh = ar[f"g_coh-{it}"]  # Get coherent Green's function
        g_cmpt = ar[f"g_cmpt-{it}"]  # Get component Green's functions
    title = "Green's Functions"

    x = np.array(list(g_coh.mesh.values()))
    fig, ax = plt.subplots()
    ax.axhline(0, color="dimgrey", lw=0.8)
    ax.axvline(0, color="dimgrey", lw=0.8)
    ax.plot(x, -g_coh[UP].imag.data[:, 0, 0], color="k", label="G$_c$")
    ax.plot(x, +g_coh[DN].imag.data[:, 0, 0], color="k")
    for i, (name, g) in enumerate(g_cmpt):
        col = f"C{i}"
        ax.plot(x, -g[UP].imag.data[:, 0, 0], lw=1.0, c=col, label="G$_{" + name + "}$")
        ax.plot(x, +g[DN].imag.data[:, 0, 0], lw=1.0, c=col)
    ax.legend()
    tit = f"{title} U={u:.1f}" if u else title
    if de:
        tit += rf" $\Delta E$={de:.1f}"
    ax.set(xmargin=0, title=tit, xlabel="ω", ylabel="Im G(ω)")
    fig.tight_layout()

    return fig, ax


def plot_self_energies(
    output_file: str, it: int = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    with HDFArchive(output_file, "r") as ar:
        params = ar["params"]
        if it is None:
            it = ar["it"]  # Get latest iteration number
        elif it < 0:
            it = ar["it"] + it
        try:
            sigma_dmft = ar[f"sigma_dmft-{it}"]  # Get DMFT self-energy
        except KeyError:
            sigma_dmft = None
        sigma_cpa = ar[f"sigma_cpa-{it}"]  # Get CPA self-energy
    title = "Self Energies"
    u = params.u
    # Plot self energies
    x = np.array(list(sigma_cpa.mesh.values()))
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.axhline(0, color="dimgrey", lw=0.8)
    ax1.axvline(0, color="dimgrey", lw=0.8)
    ax2.axhline(0, color="dimgrey", lw=0.8)
    ax2.axvline(0, color="dimgrey", lw=0.8)
    ax1.plot(x, sigma_cpa[UP].imag.data[:, 0, 0], color="k", label="CPA")
    ax2.plot(x, sigma_cpa[DN].imag.data[:, 0, 0], color="k")
    if sigma_dmft is not None:
        for cmpt, sig in sigma_dmft:
            ax1.plot(x, sig[UP].imag.data[:, 0, 0], lw=1.0, label=f"DMFT-{cmpt}")
            ax2.plot(x, sig[DN].imag.data[:, 0, 0], lw=1.0)
    ax1.legend()
    tit = f"{title} U={u}" if u else title
    ax1.set(xmargin=0, title=tit, ylabel="Im Σ$_{↑}$(ω)")
    ax2.set(xmargin=0, xlabel="ω", ylabel="Im Σ$_{↓}$(ω)")
    fig.tight_layout()

    return fig, (ax1, ax2)
