# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-09-12

import functools
import subprocess
import sys
from pathlib import Path
from typing import List

import click

from model_dmft import Folder, InputParameters, walkdirs
from model_dmft.utility import WorkingDir

__all__ = ["cli", "get_dirs", "single_path_opts", "multi_path_opts", "frmt_file"]

REMOTE = "git+ssh://git@github.com/dylanljones/model_dmft.git"
# REMOTE = "git+ssh://git@git.rz.uni-augsburg.de/jonesdyl/model_dmft"


def error(s):
    return click.style(s, fg="red")


def get_dirs(*paths, recursive=False) -> List[Folder]:
    if not paths:
        paths = (".",)
    try:
        folders = list(walkdirs(*paths, recursive=recursive))
    except Exception as e:
        raise click.ClickException(click.style(str(e), bg="red"))
    if not folders:
        raise click.ClickException(click.style("No TRIQS-CPA+DMFT directories found.", bg="red"))
    # Recursively sort the folders by path paths
    maxparts = max([len(f.path.parts) for f in folders])
    for i in reversed(range(maxparts)):
        folders = sorted(folders, key=lambda f: f.path.parts[i] if len(f.path.parts) > i else "")
    return folders


def single_path_opts(func):
    """Click argument decorator for commands accepting a single input path."""

    @click.argument("path", type=click.Path(), nargs=1, required=False, default=".")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def multi_path_opts(func):
    """Click argument decorator for commands accepting multiple input paths."""

    @click.option(
        "--recursive",
        "-r",
        is_flag=True,
        default=False,
        help="Recursively search for EMTO directories.",
    )
    @click.argument("paths", type=click.Path(), nargs=-1, required=False)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def frmt_file(s):
    return click.style(str(s), fg="magenta")


# -- CLI -------------------------------------------------------------------------------------------


@click.group(name="cpa_dmft")
def cli():
    pass


@cli.command(name="update", help="Update the application")
def update():
    import os

    cmd = f"pip install {REMOTE}"
    click.echo("Updating application")
    click.echo(f"> {cmd}")
    click.echo()
    os.system(cmd)


@cli.command(name="solve_impurity")
@click.argument("tmp_file", type=click.Path(exists=True))
def solve_impurity_cmd(tmp_file: str):
    """Solve the impurity problem."""
    from model_dmft.solver import solve_impurity

    solve_impurity(tmp_file)
    sys.exit(0)  # make sure exit code is 0 on success


@cli.command(name="run")
@click.argument(
    "file",
    type=click.Path(exists=True),
    default="inp.toml",
)
@click.option("--n_procs", "-n", default=0, type=int, help="Total number of processes to use")
def run_cmd(file: str, n_procs: int):
    """Run a CPA+DMFT calculation."""
    from model_dmft.solver import solve

    file = Path(file)
    if file.is_dir():
        # Assume the input file 'inp.toml' is in the directory
        file = file / "inp.toml"

    if not file.exists():
        raise click.ClickException(click.style(f"No input file '{file}' found!", bg="red"))

    params = InputParameters(file)
    # params.resolve_location(file.parent)
    solve(params, n_procs=n_procs)


# -- Utility ---------------------------------------------------------------------------------------


@cli.command(name="walk")
@multi_path_opts
def walk_cmd(recursive: bool, paths: List[str]):
    """Walk through directories and search for CPA+DMFT calculations.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        path = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        click.echo(f"{path} {folder.params}")


@cli.command(name="get")
@click.argument("key", type=str, nargs=1)
@multi_path_opts
def get_cmd(key: str, recursive: bool, paths: List[str]):
    """Gets the given value from the input files in the given directories.

    KEY: The key of the value to get.
    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        path = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        params = folder.params
        click.echo(f"{path} {key}={params[key]}")


@cli.command(name="set")
@click.argument("value", type=str, nargs=1)
@multi_path_opts
def set_cmd(value: str, recursive: bool, paths: List[str]):
    """Sets the given value in the input files in the given directories.

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
        click.echo(f"{path} Setting {key} to {val}")
        params = folder.params
        params[key] = val
        params.dump()


@cli.command(name="diff")
@multi_path_opts
def diff_cmd(recursive: bool, paths: List[str]):
    """Prints the differences of the given input files.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    # maxw = max(len(str(folder.path)) for folder in folders) + 1
    if len(folders) < 2:
        raise click.ClickException(click.style("Need at least two directories to compare.", bg="red"))
    elif len(folders) == 2:
        click.echo(f"Comparing {folders[0].path} and {folders[1].path}")
    else:
        raise click.ClickException(click.style("Can only compare two directories for now.", bg="red"))


# noinspection PyShadowingBuiltins
@cli.command(name="iter")
@multi_path_opts
def iter_cmd(recursive: bool, paths: List[str]):
    """Greps for the iteration number in the output files in the given directories.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        if Path(folder.params.output_path).exists():
            with folder.archive() as ar:
                if "it" not in ar:
                    msg = click.style("No iterations!", fg="red")
                else:
                    it = ar["it"]
                    msg = f"Last Iteration: {it}"
        else:
            msg = click.style("No output file!", fg="red")
        click.echo(f"{p} " + msg)


# noinspection PyShadowingBuiltins
@cli.command(name="error")
@click.option("--all", "-a", is_flag=True, default=False, help="Show all line")
@multi_path_opts
def error_cmd(all: bool, recursive: bool, paths: List[str]):
    """Greps for the error in the output files in the given directories.

    ALL: Show all iterations. The default is False (only last iteration).
    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    click.echo("")
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        s = "Error-G: {g:.10f} Error-Σ: {s:.10f} Error-n: {n:.10f}"
        if not Path(folder.params.output_path).exists():
            click.echo(f"{p}  " + click.style("No output file!", fg="red"))
        else:
            with folder.archive() as ar:
                if "it" not in ar:
                    click.echo(f"{p}  " + click.style("No iterations!", fg="red"))
                    continue
                max_it = ar["it"]
                if not all:
                    err_g = ar["err_g"]
                    err_s = ar["err_sigma"]
                    err_n = ar["err_occ"]
                    click.echo(f"{p}  [{max_it:<2}] {s.format(g=err_g, s=err_s, n=err_n)}")
                else:
                    click.echo(p)
                    for it in range(1, max_it + 1):
                        err_g = ar[f"err_g-{it}"]
                        err_s = ar[f"err_sigma-{it}"]
                        err_n = ar[f"err_occ-{it}"]
                        click.echo(f"  [{it:<2}] {s.format(g=err_g, s=err_s, n=err_n)}")
    click.echo("")


# noinspection PyShadowingBuiltins
@cli.command(name="clean")
@multi_path_opts
def clean_cmd(recursive: bool, paths: List[str]):
    """Remove all output files in the given directories.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    click.echo("")
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        click.echo(f"{p} Cleaning directory")
        folder.clear()
    click.echo("")


# noinspection PyShadowingBuiltins
@cli.command(name="clean-tmp")
@multi_path_opts
def clean_tmp(recursive: bool, paths: List[str]):
    """Remove the temporary files in the given directories.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        click.echo(f"{p} Cleaning directory")
        folder.remove_tmp_dirs()


# noinspection PyShadowingBuiltins
@cli.command(name="submit")
@click.option("--clean", "-c", is_flag=True, default=False, help="Clean old slurm files")
@multi_path_opts
def submit_cmd(clean: bool, recursive: bool, paths: List[str]):
    """Batch-run the simulations in the given directories using SLURM.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        slurm = folder.path / "run.slurm"
        if not slurm:
            click.echo(f"{p} No slurm file found")
            continue
        if clean:
            folder.remove_slurm_outputs(keep_last=False)
        with WorkingDir(folder.path):
            cmd = f"sbatch {slurm.name}"
            stdout = subprocess.check_output(cmd, shell=True)
            stdout = stdout.decode("utf-8").replace("\n", "")
            click.echo(f"{p} {stdout}")


# noinspection PyShadowingBuiltins
@cli.command(name="cancel")
@multi_path_opts
def cancel_cmd(recursive: bool, paths: List[str]):
    """Cancel the simulations in the given directories using SLURM.

    RECURSIVE: Search directories recursively. The default is False.
    PATHS: One or multiple paths to search for calculation directories. The default is '.'.
    """
    folders = get_dirs(*paths, recursive=recursive)
    maxw = max(len(str(folder.path)) for folder in folders) + 1
    for folder in folders:
        p = frmt_file(f"{str(folder.path) + ':':<{maxw}}")
        slurm_files = folder.get_slurm_outputs()
        if not slurm_files:
            click.echo(f"{p} No slurm outputs found!")
            continue
        job_ids = sorted([int(p.stem.replace("slurm-", "")) for p in slurm_files])
        last_id = job_ids[-1]
        cmd = f"scancel {last_id}"

        try:
            stdout = subprocess.check_output(cmd, shell=True)
            stdout = stdout.decode("utf-8").replace("\n", "")
            s = f"{p} Job {last_id} cancelled"
            if stdout:
                s += f" ({stdout})"
            click.echo(s)
        except subprocess.CalledProcessError:
            click.echo(f"{p} {error('Job not found')}")
