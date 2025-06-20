# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-06-20

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, run
from typing import Union

from triqs.version import version as triqs_version

BUILD_DIR = Path(".triqs")


@contextmanager
def changedir(path: Union[str, Path]) -> None:
    """Context manager to change the current working directory."""
    original_dir = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


@dataclass
class TriqsApp:
    name: str
    package: str
    repo: str
    triqs_version: str
    build: bool = True
    conda_cmd: str = None


REPOS = {
    "hartree-fock": "https://github.com/TRIQS/hartree_fock",
    "hubbardI": "https://github.com/TRIQS/hubbardI",
    "cthyb": "https://github.com/TRIQS/cthyb",
    "ctseg": "https://github.com/TRIQS/ctseg",
    "forktps": "",
}

APPS = {
    "hartree-fock": TriqsApp(
        name="hartree_fock",
        package="triqs_hartree_fock",
        repo="https://github.com/TRIQS/hartree_fock",
        triqs_version="3.3",
        build=False,
    ),
    "hubbardI": TriqsApp(
        name="hubbardI",
        package="triqs_hubbardI",
        repo="https://github.com/TRIQS/hubbardI",
        triqs_version="3.3",
    ),
    "cthyb": TriqsApp(
        name="cthyb",
        package="triqs_cthyb",
        repo="https://github.com/TRIQS/cthyb",
        triqs_version="3.3",
        conda_cmd="conda install -c conda-forge triqs_cthyb -y",
    ),
    "ctseg": TriqsApp(
        name="ctseg",
        package="triqs_ctseg",
        repo="https://github.com/TRIQS/ctseg",
        triqs_version="3.3",
        conda_cmd="conda install -c conda-forge triqs_ctseg -y",
    ),
    "forktps": TriqsApp(
        name="forktps",
        package="forktps",
        repo="https://github.com/TRIQS/forktps",
        triqs_version="3.1",
    ),
    "triqs_cpa": TriqsApp(
        name="triqs_cpa",
        package="triqs_cpa",
        repo="https://github.com/dylanljones/triqs_cpa",
        triqs_version="3.1",
        build=False,
    ),
}


def shell_source(script: str) -> None:
    """Emulate the action of "source" in bash, settings some environment variables."""
    import os
    import subprocess

    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict((line.split(b"=", 1) for line in output.splitlines()))

    for key in list(env.keys()):
        val = env[key]
        key = key.decode("utf-8") if isinstance(key, bytes) else key
        # Convert bytes to str if necessary
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        # Update the environment variable
        os.environ[key] = val


def install_from_source(app: TriqsApp) -> None:
    dirname = BUILD_DIR / app.name
    dirname.mkdir(exist_ok=True, parents=True)
    # Clone the repository if it doesn't exist
    with changedir(dirname):
        outdir = Path(f"{app.name}.src")
        if not outdir.exists():
            print(f"Cloning {app.name} from {app.repo} into {outdir}")
            cmd = f"git clone {app.repo} {outdir.name}"
            result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
            print(result.stdout)

    build_dir = dirname / f"{app.name}.build"
    build_dir.mkdir(exist_ok=True, parents=True)
    with changedir(build_dir):
        # Source the triqsvars.sh script to set up the environment
        shell_source(f"{os.environ['CONDA_PREFIX']}/share/triqs/triqsvars.sh")
        # Run cmake to configure the build
        print(f"Configuring {app.name} in {build_dir}")
        cmd = f"cmake ../{app.name}.src"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            print(f"Error running command '{cmd}': {result.stderr}")
            return
        print(result.stdout)

        if app.build:
            # Build the package
            print(f"Building {app.name} in {build_dir}")
            cmd = "make"
            result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
            if result.returncode != 0:
                print(f"Error running command '{cmd}': {result.stderr}")
                return
            print(result.stdout)

        # Install the package
        print(f"Installing {app.name} in {build_dir}")
        cmd = "make install"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            print(f"Error running command '{cmd}': {result.stderr}")
            return
        print(result.stdout)


def install_from_conda(app: TriqsApp) -> None:
    cmd = app.conda_cmd
    result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running command '{cmd}': {result.stderr}")
        return
    print(result.stdout)


def install_app(app: TriqsApp) -> None:
    if not triqs_version.startswith(app.triqs_version):
        print(f"{app.name} not compatible with TRIQS version {triqs_version}.")
        return

    try:
        __import__(app.package)
        print(f"{app.name} is already installed.")
        return
    except ImportError:
        pass

    if app.conda_cmd:
        install_from_conda(app)
    else:
        install_from_source(app)


def main() -> None:
    for name, app in APPS.items():
        print(f"Installing {name}...")
        install_app(app)
        print()


if __name__ == "__main__":
    main()
