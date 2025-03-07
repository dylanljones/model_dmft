# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-02-12

from pathlib import Path

import numpy as np

# noinspection PyPackageRequirements
from h5 import HDFArchive
from triqs.gf import BlockGf

from .input import InputParameters
from .utility import mesh_to_array

__all__ = ["write_out_files"]


def _write_gf(
    g_coh: BlockGf, g_cmpt: BlockGf, location: Path, frmt: str, is_real_mesh: bool
) -> None:
    suffix = "_w" if is_real_mesh else "_iw"
    freq_name = "w" if is_real_mesh else "iw"
    omega = mesh_to_array(g_coh.mesh)
    # Write Gf file
    names, items = list(), list()
    names.append(freq_name)
    items.append(omega)
    for spin, g in g_coh:
        names.append(f"Re G({spin})")
        names.append(f"Im G({spin})")
        items.append(g.data[:, 0, 0].real)
        items.append(g.data[:, 0, 0].imag)
    # Components
    for name, gf in g_cmpt:
        for spin, g in gf:
            names.append(f"Re G({name}-{spin})")
            names.append(f"Im G({name}-{spin})")
            items.append(g.data[:, 0, 0].real)
            items.append(g.data[:, 0, 0].imag)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / f"gf{suffix}.dat", data, header=header, fmt=frmt, delimiter="  ")


def _write_sigma_cpa(sigma_coh: BlockGf, location: Path, frmt: str, is_real_mesh: bool) -> None:
    suffix = "_w" if is_real_mesh else "_iw"
    freq_name = "w" if is_real_mesh else "iw"
    omega = mesh_to_array(sigma_coh.mesh)
    # Write coherent self-energy file
    names, items = list(), list()
    names.append(freq_name)
    items.append(omega)
    for spin, sig in sigma_coh:
        names.append(f"Re Sig({spin})")
        names.append(f"Im Sig({spin})")
        items.append(sig.data[:, 0, 0].real)
        items.append(sig.data[:, 0, 0].imag)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / f"sigma_coh{suffix}.dat", data, header=header, fmt=frmt, delimiter="  ")


def _write_sigma_dmft(sigma_dmft: BlockGf, location: Path, frmt: str, is_real_mesh: bool) -> None:
    suffix = "_w" if is_real_mesh else "_iw"
    freq_name = "w" if is_real_mesh else "iw"
    omega = mesh_to_array(sigma_dmft.mesh)
    # Write DMFT self-energy file
    names, items = list(), list()
    names.append(freq_name)
    items.append(omega)
    for name, sigma in sigma_dmft:
        for spin, sig in sigma:
            names.append(f"Re SIG({name}-{spin})")
            names.append(f"Im SIG({name}-{spin})")
            items.append(sig.data[:, 0, 0].real)
            items.append(sig.data[:, 0, 0].imag)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / f"sigma_dmft{suffix}.dat", data, header=header, fmt=frmt, delimiter="  ")


def _write_dos(
    g_coh: BlockGf,
    g_cmpt: BlockGf,
    location: Path,
    frmt: str,
) -> None:
    omega = mesh_to_array(g_coh.mesh)
    names, items = list(), list()
    names.append("omega")
    items.append(omega)
    for spin, g in g_coh:
        names.append(f"DOS({spin})")
        items.append(-g.data[:, 0, 0].imag / np.pi)
    # Components
    for name, gf in g_cmpt:
        for spin, g in gf:
            names.append(f"PDOS({name}-{spin})")
            items.append(-g.data[:, 0, 0].imag / np.pi)
    header = "   ".join(names)
    data = np.array(items).T
    np.savetxt(location / "dos.dat", data, header=header, fmt=frmt, delimiter="  ")


def write_out_files(params: InputParameters) -> None:
    """Write main quantities to plain text output files.

    Parameters
    ----------
    params : InputParameters
        The input parameters.
    """
    frmt = "%+.16f"
    location = Path(params.location)
    archive_file = str(location / params.output)

    # Load data from archive
    with HDFArchive(archive_file, "r") as ar:
        g_coh = ar["g_coh"]
        g_cmpt = ar["g_cmpt"]
        sigma_coh = ar["sigma_cpa"]
        sigma_dmft = ar["sigma_dmft"] if "sigma_dmft" in ar else None

        # Anacont results
        g_coh_real = ar["g_coh_real"] if "g_coh_real" in ar else None
        g_cmpt_real = ar["g_cmpt_real"] if "g_cmpt_real" in ar else None
        sigma_cpa_real = ar["sigma_cpa_real"] if "sigma_cpa_real" in ar else None

    is_real = params.is_real_mesh
    # Write main output files
    _write_gf(g_coh, g_cmpt, location, frmt, is_real)
    _write_sigma_cpa(sigma_coh, location, frmt, is_real)
    if sigma_dmft is not None:
        _write_sigma_dmft(sigma_dmft, location, frmt, is_real)

    # Write analytical continuation output files
    if g_coh_real is not None and g_cmpt_real is not None:
        _write_gf(g_coh_real, g_cmpt_real, location, frmt, is_real_mesh=True)
    if sigma_cpa_real is not None:
        _write_sigma_cpa(sigma_cpa_real, location, frmt, is_real_mesh=True)

    # Write DOS
    if is_real:
        _write_dos(g_coh, g_cmpt, location, frmt)
    elif g_coh_real is not None and g_cmpt_real is not None:
        _write_dos(g_coh_real, g_cmpt_real, location, frmt)
