# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-15

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Iterable, List, Union

import tomlkit

from .archive import Archive
from .input import InputParameters

__all__ = ["Folder", "walkdirs"]


def find_input_file(root: Union[str, Path], check_content: bool = True) -> Union[Path, None]:
    """Search for an input file in the given directory.

    The input file must be a .toml file and contain at least the section "general" with the
    "output" key. If multiple input files are found, an error is raised. If no input file is found,
    None is returned.
    """
    # Find .toml files
    files = list(Path(root).glob("*.toml"))
    if len(files) == 0:
        # raise FileNotFoundError(f"No toml files found in {root}")
        return None

    # Check if the toml file has the expected sections
    candidates = list()
    for file in files:
        if check_content:
            with open(file, "r") as f:
                text = f.read()
            data = tomlkit.parse(text)
            if "general" in data:
                candidates.append(file)
        else:
            candidates.append(file)

    if len(candidates) == 0:
        # raise FileNotFoundError(f"No input file found in {root}")
        return None
    if len(candidates) > 1:
        raise FileNotFoundError(f"Multiple input files found in {root}")

    return candidates[0]


def find_slurm_scripts(root: Union[str, Path]) -> Path:
    """Search for SLURM scripts in the given directory.

    The SLURM scripts must have the extension .slurm.
    """
    candidates = list(Path(root).glob("*.slurm"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No SLURM scripts found in {root}")
    elif len(candidates) > 1:
        raise FileNotFoundError(f"Multiple SLURM scripts found in {root}")
    return candidates[0]


class Folder:
    def __init__(
        self,
        *path: Union[str, Path],
        input_file: Union[str, Path] = None,
        assert_exists: bool = False,
    ):
        self.path = Path(*path)
        self._input_file = input_file
        self._params = None
        self._slurm_file = None
        if assert_exists:
            self.assert_exists()

    @property
    def input_file(self) -> Path:
        if self._input_file is None:
            self._input_file = find_input_file(self.path)
        return self._input_file

    @property
    def output_file(self) -> Path:
        file = self.path / self.params.output
        # if not file.exists():
        #     raise FileNotFoundError(f"Output file '{file}' not found!")
        return file

    @property
    def slurm_path(self) -> Path:
        if self._slurm_file is None:
            try:
                self._slurm_file = find_slurm_scripts(self.path)
            except FileNotFoundError as e:
                print(e)
        return self._slurm_file

    @property
    def params(self) -> InputParameters:
        if self._params is None:
            self._params = InputParameters(self.input_file)
        return self._params

    def exists(self) -> bool:
        """Check if the folder exists."""
        return self.path.exists()

    def assert_exists(self) -> None:
        """Assert that the folder exists."""
        if not self.exists():
            raise FileNotFoundError(f"Folder '{self.path}' does not exist!")

    def get_slurm_outputs(self) -> List[Path]:
        return [Path(p) for p in sorted(self.path.glob("slurm-*.out"))]

    def read_slurm_output(self) -> str:
        """Read the last slurm output file."""
        files = self.get_slurm_outputs()
        if files:
            with open(files[-1], "r") as f:
                return f.read()
        return ""

    def remove_slurm_outputs(self, keep_last: bool = False) -> None:
        files = sorted(self.get_slurm_outputs())
        if keep_last and files:
            files.pop(-1)
        for file in files:
            file.unlink()

    def remove_tmp_dirs(self) -> None:
        shutil.rmtree(self.params.tmp_dir_path, ignore_errors=True)

    def remove_archive_files(self) -> None:
        for file in self.path.glob("*.h5"):
            file.unlink()

    def remove_output_files(self) -> None:
        for file in self.path.glob("*.dat"):
            file.unlink()

    def remove_log_files(self) -> None:
        for file in self.path.glob("*.log"):
            file.unlink()

    def clear(self, slurm: bool = True, logs: bool = True, data: bool = True, tmp: bool = True) -> None:
        if slurm:
            self.remove_slurm_outputs()
        if data:
            self.remove_archive_files()
            self.remove_output_files()
        if logs:
            self.remove_log_files()
        if tmp:
            self.remove_tmp_dirs()

    @contextmanager
    def archive(self, mode: str = "r") -> ContextManager[Archive]:
        file = self.output_file
        with Archive(str(file), mode) as ar:
            yield ar

    def load_output(self, *keys, it: int = -1) -> dict:
        data = dict()
        with self.archive("r") as ar:
            if it < 0:
                it = ar["it"]
            data["it"] = it
            for key in keys:
                data[key] = ar[f"{key}-{it}"]
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"


def walkdirs(*paths: Union[str, Path], recursive: bool = False, check: bool = True) -> Iterable[Folder]:
    """Walk through directories and yield project folders."""
    paths = paths or (".",)  # Use current directory if no path is given
    for root in paths:
        root = Path(root)
        if root.is_dir():
            try:
                # Check if an input file is present in the current directory
                input_file = find_input_file(root, check_content=check)
                if input_file is not None:
                    # Found an input file, this is a project folder.
                    yield Folder(root, input_file=input_file)
            except FileNotFoundError:
                pass

        # Iterate (recursively) over all folders in the given root path
        iterator = root.rglob("*") if recursive else root.glob("*")
        for folder in iterator:
            if folder.is_dir():
                try:
                    # Check if an input file is present in the current directory
                    input_file = find_input_file(folder, check_content=check)
                    if input_file is not None:
                        # Found an input file, this is a project folder.
                        yield Folder(folder, input_file=input_file)
                except FileNotFoundError:
                    pass
