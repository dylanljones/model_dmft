# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-09-12

import sys
from pathlib import Path

import click

from model_dmft import InputParameters

from .cli import cli


@cli.group(name="solve")
def _solve():
    """Run CPA+DMFT calculations or solvers."""
    pass


@cli.command(name="solve_impurity")
@click.argument("tmp_file", type=click.Path(exists=True))
def _solve_impurity(tmp_file: str):
    """Solve the impurity problem."""
    from model_dmft.runner import solve_impurity

    solve_impurity(tmp_file)
    sys.exit(0)  # make sure exit code is 0 on success


@cli.command(name="run")
@click.argument(
    "file",
    type=click.Path(exists=True),
    default="inp.toml",
)
@click.option("--n_procs", "-n", default=0, type=int, help="Total number of processes to use")
def _solve_(file: str, n_procs: int):
    """Run a CPA+DMFT calculation."""
    from model_dmft.runner import solve

    file = Path(file)
    params = InputParameters(file)
    params.resolve_location(file.parent)
    solve(params, n_procs=n_procs)
