# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-09-12

from typing import List

import click
from slurmio import SlurmScript

from .cli import cli, frmt_file, get_dirs, multi_path_opts


@cli.group(name="slurm")
def _slurm():
    """Manage SLURM scripts for CPA+DMFT calculations."""
    pass


@_slurm.command(name="get")
@click.argument("key", type=str, nargs=1)
@multi_path_opts
def get_cmd(key: str, recursive: bool, paths: List[str]):
    """Gets the given value from the SLURM scripts in the given directories.

    KEY: The key of the value to get.
    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        path = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        slurm_path = folder.slurm_path
        if slurm_path is None or not slurm_path.exists():
            click.echo(f"{path} No SLURM script found.")
            continue
        slurm = SlurmScript(slurm_path)
        click.echo(f"{path} {key}={slurm.options[key]}")


@_slurm.command(name="set")
@click.argument("value", type=str, nargs=1)
@multi_path_opts
def set_cmd(value: str, recursive: bool, paths: List[str]):
    """Sets the given value in the SLURM scripts in the given directories.

    VALUE: The key of the value to set. Must be in the form KEY=VALUE.
    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    key, val = value.split("=")
    key, val = key.strip(), val.strip()
    for folder in folders:
        path = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        slurm_path = folder.slurm_path
        if slurm_path is None or not slurm_path.exists():
            click.echo(f"{path} No SLURM script found.")
            continue
        slurm = SlurmScript(slurm_path)
        click.echo(f"{path} Setting {key} to {val}")
        slurm.options[key] = val
        slurm.dump()
