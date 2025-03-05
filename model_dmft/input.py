# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-08

"""This module defines the `InputParameters` class for handling input parameters for TRIQS CPA+DMFT.

Classes
-------
Parameters : Base class for handling parameter mappings.
SolverParams : Base class for solver-specific parameters.
FtpsSolverParams : Class for handling FTPS solver parameters.
CthybSolverParams : Class for handling CTHYB solver parameters.
InputParameters : Class for handling general input parameters for the simulation.

Functions
---------
register_solver_input(cls)
    Registers a solver input class.
"""

import os
from abc import ABC
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import tomlkit as toml
from h5.archive import register_class
from triqs.gf import MeshImFreq, MeshReFreq

__all__ = [
    "InputParameters",
    "FtpsSolverParams",
    "CthybSolverParams",
    "HubbardISolverParams",
    "HartreeSolverParams",
    "SolverParams",
    "MaxEntParams",
    "PadeParams",
    "get_supported_solvers",
]

SOLVERS: Dict[str, Type["SolverParams"]] = dict()
LATTICES = ("bethe", "square")
HALF_BANDWIDTHS = {
    "bethe": 1.0,
    "square": 2.0,
}


def _indent_comments(lines: List[str]) -> List[str]:
    """Indent the comments to align in width."""

    def is_commented_line(s: str) -> bool:
        if s.strip().startswith("["):
            return False  # Section header
        if s.strip().startswith("#"):
            return False  # Full comment line
        return "#" in s

    # Find the maximum length of the text before the comment for indentation
    try:
        w = max(len(s.split("#", 1)[0].strip()) for s in filter(is_commented_line, lines))
    except ValueError:
        return lines

    # Format the lines with comments
    for i, line in enumerate(lines):
        if is_commented_line(line):
            text, comment = line.split("#", 1)
            lines[i] = f"{text.strip():<{w}}  # {comment.strip()}"

    return lines


def _indent_values(lines: List[str]) -> List[str]:
    """Indent the comments to align in width."""

    def is_value_line(s: str) -> bool:
        if s.strip().startswith("#"):
            return False  # Full comment line
        return "=" in s

    # Find the maximum length of the text before the comment for indentation
    try:
        w = max(len(s.split("=", 1)[0].strip()) for s in filter(is_value_line, lines))
    except ValueError:
        return lines
    # Format the lines with comments
    for i, line in enumerate(lines):
        if is_value_line(line):
            name, rest = line.split("=", 1)
            lines[i] = f"{name.strip():<{w}} = {rest.strip()}"

    return lines


def register_solver_input(cls: Type["SolverParams"]) -> None:
    """Register a valid solver input class."""
    SOLVERS[cls.SOLVER] = cls


def get_supported_solvers() -> List[str]:
    """Return a list of supported solvers."""
    return list(SOLVERS.keys())


def _parse_array(value: Union[str, float, Sequence[float]]) -> Union[float, Sequence[float]]:
    if isinstance(value, str):
        if "," in value:
            value = value.strip("[]")
            return list(float(x) for x in value.split(","))
        return float(value)
    return value


class InputError(ValueError):
    pass


class InputMeshError(InputError):
    def __init__(self, mesh_type, param_name):
        super().__init__(f"Mesh type '{mesh_type}' requires parameter '{param_name}' to be set!")


class Parameters(Mapping, ABC):
    """Base class for handling parameter mappings.

    This class provides a dictionary-like interface for managing parameters, including methods
    for updating, comparing, and converting parameters to different formats.
    """

    __types__ = dict()
    __descriptions__ = dict()

    def __init__(self, **kwargs):
        self.update(kwargs)

    def __iter__(self):
        return filter(lambda k: not k.startswith("_"), self.__dict__)

    def __len__(self) -> int:
        return len(tuple(self.__iter__()))

    def __getitem__(self, key: str) -> Any:
        return self.__getattribute__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.__types__:
            value = self.__types__[key](value)
        self.__setattr__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __eq__(self, other: Union["Parameters", Dict[str, Any]]) -> bool:
        if isinstance(other, Parameters):
            other = other.dict(include_none=False)
        return self.dict(include_none=False) == other

    def diff(self, other: Union["Parameters", Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(other, Parameters):
            other = other.dict(include_none=False)
        diffs = dict()
        for key in self.keys():
            if isinstance(self[key], Parameters):
                diff = self[key].diff(other[key])
                if diff:
                    diffs[key] = diff
            else:
                if self[key] != other.get(key, None):
                    diffs[key] = (self[key], other[key])
        return diffs

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            if not hasattr(self, key):
                print(f"WARNING: Setting unknown input parameter '{key}'!")
            self.__setitem__(key, value)

    def dict(self, include_none: bool = True) -> Dict[str, Any]:
        data = dict()
        for k, v in self.items():
            if v is not None or include_none:
                if isinstance(v, Parameters):
                    v = v.dict(include_none)
                data[k] = v
        return data

    @classmethod
    def __factory_from_dict__(cls, name: str, d: Dict[str, Any]) -> "Parameters":
        self = cls()
        self.update(d)
        return self

    def __reduce_to_dict__(self) -> Dict[str, Any]:
        return self.dict()


# -- Solver input parameters -----------------------------------------------------------------------


class SolverParams(Parameters):
    """Base class for solver-specific parameters.

    This class extends the `Parameters` class to include attributes and methods specific to solver
    parameters.
    """

    SOLVER: str
    RE_MESH: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type: str = self.SOLVER

    def __repr__(self):
        return f"{self.__class__.__name__}({self.type})"


class FtpsSolverParams(SolverParams):
    """Class for handling FTPS solver parameters.

    This class extends the `SolverParams` class to include attributes and methods specific
    to the FTPS solver.
    """

    SOLVER = "ftps"
    RE_MESH = True

    __types__ = {
        "n_bath": int,
        "time_steps": int,
        "dt": float,
        "sweeps": int,
        "tw": float,
        "maxm": int,
        "nmax": int,
    }

    __descriptions__ = {
        "bath_fit": "DiscretizeBath vs BathFitter.",
        "n_bath": "The number of bath sites used by the FTPS solver.",
        "time_steps": "Number of time steps for the time evolution.",
        "dt": "Time step for the time evolution.",
        "method": 'Time evolution method, "TDVP", "TDVP_2" or "TEBD".',
        "sweeps": "Number of DMRG sweeps (default 15).",
        "tw": "Truncated weight for every link. (default 1e-9).",
        "maxm": "Maximum bond dimension. (default 100).",
        "nmax": "Maximal Number of Krylov vectors created. (default 40).",
    }

    def __init__(self, **kwargs):
        # General
        self.bath_fit: bool = True  # DiscretizeBath vs BathFitter
        self.n_bath: int = 64  # Number of bath sites.

        # Solve parameters
        # Path where to store temporary files for tevo states, default is '.tmp/'.
        # self.state_storage: Optional[str] = ".tmp/"

        # Tevo parameters
        self.time_steps: int = 100  # Number of time steps for the time evolution.
        self.dt: float = 0.1  # Time step for the time evolution.
        self.method: Optional[str] = "TDVP_2"  # Time evolution method, "TDVP", "TDVP_2" or "TEBD"

        # DMRG parameters
        self.sweeps: Optional[int] = 15  # Number of DMRG sweeps (default 15)

        # Used for Tevo and DMRG parameters
        self.tw: Optional[float] = 1e-9  # Truncated weight for every link. (default 1e-9)
        self.maxm: Optional[int] = 100  # Maximum bond dimension. (default 100)
        self.nmax: Optional[int] = 40  # Maximal Number of Krylov vectors created. (default 40)

        super().__init__(**kwargs)


class CthybSolverParams(SolverParams):
    """
    Class for handling CTHYB solver parameters.

    This class extends the `SolverParams` class to include attributes and methods specific
    to the CTHYB solver.
    """

    SOLVER = "cthyb"
    RE_MESH = False

    __types__ = {
        "n_cycles": int,
        "length_cycle": int,
        "n_warmup_cycles": int,
        "n_tau": int,
        "tail_fit": bool,
        "fit_max_moment": int,
        "fit_min_n": int,
        "fit_max_n": int,
        "measure_g_l": bool,
        "n_l": int,
        "density_matrix": bool,
    }

    __descriptions__ = {
        "n_cycles": "Number of Quantum Monte Carlo cycles (default 10_000)",
        "length_cycle": "Length of the cycle (default: 100)",
        "n_warmup_cycles": "Number of warmup cycles (default: 1_000)",
        "n_tau": "Number of imaginary time steps. (default: 10001)",
        "tail_fit": "Perform tail fit of Sigma and G (default: false)",
        "fit_max_moment": "Highest moment to fit in the tail of Sigma (default: 3)",
        "fit_min_n": "Index of iw from which to start fitting (default: 0.5*n_iw)",
        "fit_max_n": "Index of iw up to which to fit (default: n_iw)",
        "measure_g_l": "Measure G_l (Legendre) (default: false)",
        "n_l": "Number of Legendre polynomials. (default: 30)",
        "density_matrix": "Measure the impurity density matrix (default: false)",
    }

    def __init__(self, **kwargs):
        # General
        self.n_cycles: int = 10_000  # Number of QMC cycles.
        self.n_warmup_cycles: int = 1_000  # Number of warmup cycles.
        self.length_cycle: int = 100  # Length of a cycle.
        self.n_tau: int = 10001  # Number of imaginary time steps.
        self.tail_fit: bool = False  # Perform tail fit.
        self.fit_max_moment: int = 3  # Highest moment to fit in the tail of Sigma
        self.fit_min_n: int = 0  # Index of iw from which to start fitting.
        self.fit_max_n: int = 0  # Index of iw up to which to fit.
        self.measure_g_l: bool = False  # Measure G_l (Legendre)
        self.n_l: int = 30  # Number of Legendre polynomials.
        self.density_matrix: bool = False  # Measure the impurity density matrix.
        super().__init__(**kwargs)


class HubbardISolverParams(SolverParams):
    """Class for handling HubbardI solver parameters.

    This class extends the `SolverParams` class to include attributes and methods specific
    to the HubbardI solver.
    """

    SOLVER = "hubbardI"
    RE_MESH = False

    __types__ = {
        "n_tau": int,
        "legendre_fit": bool,
        "measure_g_tau": bool,
        "measure_g_l": bool,
        "n_l": int,
        "density_matrix": bool,
    }

    __descriptions__ = {
        "n_tau": "Number of imaginary time steps. (default: 10001)",
        "legendre_fit": "filter noise of G(tau) with G_l from n_l (default: false)",
        "measure_g_l": "Measure G_l (Legendre) (default: false)",
        "measure_g_tau": "Measure G_tau (default: false)",
        "n_l": "Number of Legendre coefficients. (default: 30)",
        "density_matrix": "Measure the impurity density matrix (default: false)",
    }

    def __init__(self, **kwargs):
        self.n_tau: int = 10001  # Number of imaginary time steps.
        self.legendre_fit: bool = False  # Filter noise of G(tau) with G_l from n_l.
        self.density_matrix: bool = False  # Measure the impurity density matrix.
        self.measure_g_tau: bool = True  # Measure G_tau
        self.measure_g_l: bool = False  # Measure G_l (Legendre)
        self.n_l: int = 30  # Number of Legendre polynomials.
        super().__init__(**kwargs)


class HartreeSolverParams(SolverParams):
    """Class for handling Hartree-Fock solver parameters.

    This class extends the `SolverParams` class to include attributes and methods specific
    to the hartree-Fock solver.
    """

    SOLVER = "hartree"
    RE_MESH = False

    __types__ = {
        "one_shot": bool,
        "method": str,
        "tol": float,
        "with_fock": bool,
        "force_real": bool,
    }

    __descriptions__ = {
        "one_shot": "Perform a one-shot or self-consitent root finding (default: False)",
        "method": "Method used for root finding (default: 'krylov')",
        "tol": "Tolerance for root finder if one_shot=False (default: 1e-5)",
        "with_fock": "Include Fock exchange terms in the self-energy (default: False)",
        "force_real": "Force the self energy from Hartree fock to be real (default: True)",
    }

    def __init__(self, **kwargs):
        self.one_shot: bool = False  # Perform a one-shot or self-consitent root finding
        self.method: str = "krylov"  # Method used for root finding
        self.tol: float = 1e-5  # tolerance for root finder if one_shot=False.
        self.with_fock: bool = False  # include Fock exchange terms in the self-energy
        self.force_real: bool = True  # force the self energy from Hartree fock to be real
        super().__init__(**kwargs)


SolversUnion = Union[FtpsSolverParams, CthybSolverParams, HubbardISolverParams, HartreeSolverParams]

# -- Maxent input parameters -----------------------------------------------------------------------


class MaxEntParams(Parameters):
    """Maximum Entropy solver parameters.

    This class extends the `Parameters` class to include attributes and methods specific to MaxEnt
    parameters.
    """

    __types__ = {
        "error": float,
        "cost_function": str,
        "probability": str,
        "mesh_type_alpha": str,
        "n_alpha": int,
        "alpha_range": _parse_array,
        "mesh_type_w": str,
        "n_w": int,
        "w_range": _parse_array,
        "probabilities": str,
    }

    __descriptions__ = {
        "error": "Error threshold (default: 1e-4)",
        "cost_function": "Cost function (default: bryan)",
        "probability": "Probability distribution (default: normal)",
        "mesh_type_alpha": "Alpha mesh type (default: logarithmic)",
        "n_alpha": "Number of alpha mesh points",
        "alpha_range": "Alpha range",
        "mesh_type_w": "Frequency mesh type (default: hyperbolic)",
        "n_w": "Number of frequency mesh points",
        "w_range": "Frequency range",
    }

    def __init__(self, **kwargs):
        self.error = 1e-4  # Error threshold for the MaxEnt solver.
        self.cost_function: Optional[str] = "bryan"  # Cost function for the MaxEnt solver.
        self.probability: Optional[str] = "normal"  # Probability distribution.
        # Alpha mesh
        self.mesh_type_alpha: Optional[str] = "logarithmic"
        self.n_alpha: Optional[int] = 60  # Number of alpha mesh points.
        self.alpha_range: Optional[Tuple[float, float]] = (0.01, 2000)  # The range of alpha values.
        # Omega_mesh
        self.mesh_type_w: Optional[str] = "hyperbolic"
        self.n_w: Optional[int] = 201  # Number of real frequencies.
        self.w_range: Optional[Tuple[float, float]] = (-10, +10)  # The range of real frequencies.

        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# -- Pade input parameters -----------------------------------------------------------------------


class PadeParams(Parameters):
    """Pade parameters.

    This class extends the `Parameters` class to include attributes and methods specific to Pade
    """

    __types__ = {
        "n_w": int,
        "w_range": _parse_array,
        "n_points": int,
        "freq_offset": float,
    }

    __descriptions__ = {
        "n_w": "Number of frequency mesh points",
        "w_range": "Frequency range",
        "n_points": "Number of points for the Pade approximation",
        "freq_offset": "Frequency offset for the Pade approximation",
    }

    def __init__(self, **kwargs):
        self.n_w: Optional[int] = 201  # Number of real frequencies.
        self.w_range: Optional[Tuple[float, float]] = (-10, +10)  # The range of real frequencies.
        self.n_points: int = 100  # Number of points for the Pade approximation.
        self.freq_offset: float = 0.0  # Frequency offset for the Pade approximation.
        super().__init__(**kwargs)

    @property
    def mesh(self) -> MeshReFreq:
        """The frequency mesh used for the pade continuation."""
        return MeshReFreq(*self.w_range, self.n_w)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# -- Input parameters ------------------------------------------------------------------------------


class InputParameters(Parameters):
    """Class for handling general input parameters for the TRIQS CPA+DMFT simulation framework.

    This class extends the `Parameters` class to include attributes and methods specific to the
    input parameters required for the TRIQS CPA+DMFT simulation.
    It also includes the parameters for the supported solvers.
    """

    __types__ = {
        "n_loops": int,
        # "restart": bool,
        # "store_iter": bool,
        "use_srun": bool,
        "load_iter": int,
        "half_bandwidth": float,
        "conc": _parse_array,
        "u": _parse_array,
        "eps": _parse_array,
        "h_field": _parse_array,
        "symmetrize": bool,
        "beta": float,
        "mu": float,
        "n_iw": int,
        "n_w": int,
        "w_range": _parse_array,
        "eta": float,
        "maxiter_cpa": int,
        "verbosity_cpa": int,
        "mixing_dmft": float,
        "mixing_cpa": float,
        "gtol": float,
        "stol": float,
        "occ_tol": float,
    }

    __descriptions__ = {
        "jobname": "The job name of the simulation.",
        "location": "The directory where the simulation is run. (default: '.')",
        "output": "The name of the output file. (default: 'out.h5')",
        "tmpdir": "The directory where temporary files are stored. (default: '.tmp/')",
        "n_loops": "The total number of iterations to perform.",
        "load_iter": "Continue from specific iteration (-1 for last iteration, 0 for restart).",
        # "restart": "Flag if the calculation should resume from previous results or start over.",
        # "store_iter": "Flag to keep intermediate iteration results.",
        "use_srun": "Use srun instead of mpirun for parallel jobs",
        "lattice": "The lattice type.",
        "gf_struct": "The structure of the Greens function.",
        "half_bandwidth": "The half bandwidth of the lattice.",
        "t": "The hopping parameter (alternative to half_bandwidth).",
        "conc": "The concentrations of the components as single value (no CPA) or list of numbers.",
        "eps": "The on-site energies of the components.",
        "h_field": "The magnetic Zeeman field of the components.",
        "beta": "The inverse temperature.",
        "symmetrize": "Symmetrize the spin channels (if no magnetic field)",
        "u": "The Hubbard interaction strength (Coulomb repulsion) of the components.",
        "mu": "The chemical potential.",
        "n_w": "Number of real mesh points.",
        "w_range": "Real frequency range.",
        "n_iw": "Number of Matsubara frequencies.",
        "eta": "Complex broadening.",
        "mixing_dmft": "Mixing of the DMFT self energy.",
        "gtol": "Convergence tolerance for the coherent Green's function.",
        "stol": "Convergence tolerance for the self energy.",
        "occ_tol": "Convergence tolerance for the occupation number.",
        "mixing_cpa": "Mixing of the CPA self energy.",
        "method_cpa": "Method used for computing the CPA self energy.",
        "maxiter_cpa": "Number of iterations for computing the CPA self energy.",
        "tol_cpa": "Tolerance for the CPA self-consistency check.",
        "verbosity_cpa": "Verbosity level of the CPA iterations.",
    }

    def __init__(
        self, path: Union[str, Path] = None, load: bool = True, solver: str = None, **kwargs
    ):
        self._path: Path = Path(path) if path is not None else None

        self._jobname: str = "TRIQS_CPA+DMFT"  # The job name. Can be a formattable string.
        self._output: str = "out.h5"  # The output file name.
        self._location: str = "."  # The directory of the calculation
        self.tmpdir: str = ".tmp/"  # Directory to store temporary files
        self.n_loops: int = 10  # Number of iterations.
        self.load_iter: int = 0  # Load iteration from which to start the simulation.
        # self.restart: bool = False  # Overwrite existing output file.
        # self.store_iter: bool = True  # Keep intermediate iteration results.
        self.use_srun: bool = False  # Use srun instead of mpirun for parallel jobs

        self.lattice: str = "bethe"  # The lattice type.
        self.gf_struct = [("up", 1), ("dn", 1)]  # The Green's function structure.
        self.half_bandwidth: float = 1.0  # The half bandwidth.
        self.conc: Union[float, Sequence[float]] = 1.0  # The concentration of the components.
        self.u: Union[float, Sequence[float]] = 0.0  # The interaction strengths of the components.
        self.eps: Union[float, Sequence[float]] = 0.0  # The on-site energies of the components.
        self.h_field: Union[float, Sequence[float]] = 0.0  # Magnetic field of the components.
        self.symmetrize: bool = False  # Symmetrize the spin channels (if no field).

        self.beta: Optional[float] = None  # Inverse temperature. If set, use a complex grid.
        # self.dc: bool = True  # Is the double-counting correction U/2 on?
        self.mu: float = 0.0  # The chemical potential.

        # Imaginary mesh
        self.n_iw: Optional[int] = None  # Number of Matsubara frequencies.
        # Real mesh
        self.n_w: Optional[int] = None  # Number of real frequencies.
        self.w_range: Optional[Tuple[float, float]] = None  # The range of real frequencies.
        self.eta: Optional[float] = 0.0  # Complex broadening. Used for real grids.

        self.method_cpa: str = "iter"  # The CPA method.
        self.maxiter_cpa: Optional[int] = 1_000  # Maximum number of iterations if method is 'iter'.
        self.verbosity_cpa: Optional[int] = 0  # Verbosity level for the CPA if method is 'iter'.

        self.mixing_dmft: float = 1.0  # Mixing parameter for the DMFT self-energy.
        self.mixing_cpa: float = 1.0  # Mixing parameter for the CPA self-energy.

        self.tol_cpa: Optional[float] = 1e-6  # Tolerance for the CPA if method is 'iter'.
        self.gtol: Optional[float] = None  # Tolerance for the coherent Green's function.
        self.stol: Optional[float] = None  # Tolerance for the self energy.
        self.occ_tol: Optional[float] = None  # Tolerance for the occupation number.

        self.solver_params: Optional[SolversUnion] = None
        self.maxent_params: Optional[MaxEntParams] = None
        self.pade_params: Optional[PadeParams] = None

        super().__init__(**kwargs)
        if solver is not None:
            self.add_solver(solver)
        if load:
            self.load()

    @property
    def jobname(self) -> str:
        """Formatted job name."""
        return self.format_jobname()

    @jobname.setter
    def jobname(self, value: str) -> None:
        self._jobname = value

    @property
    def output(self) -> str:
        """Formatted output file name."""
        return self.format_output()

    @output.setter
    def output(self, value: str) -> None:
        self._output = value

    @property
    def location(self) -> str:
        """Formatted location path."""
        return self.format_location()

    @location.setter
    def location(self, value: Union[str, Path]) -> None:
        self._location = str(value)

    @property
    def location_path(self) -> str:
        """The absolute path to the location."""
        return str(Path(self.location).resolve())

    @property
    def output_path(self) -> str:
        """The absolute path to the output file."""
        return str(Path(self.location).resolve() / self.output)

    @property
    def tmp_dir_path(self) -> str:
        """The absolute path to the temporary directory."""
        return str(Path(self.location).resolve() / self.tmpdir)

    @property
    def solver(self) -> str:
        """The name of the solver."""
        return self.solver_params.type if self.solver_params else None

    @property
    def t(self) -> float:
        """The hopping parameter. This is an alternative to `half_bandwidth`."""
        return self.half_bandwidth / HALF_BANDWIDTHS[self.lattice]

    @t.setter
    def t(self, value: float) -> None:
        self.half_bandwidth = value * HALF_BANDWIDTHS[self.lattice]

    @property
    def is_real_mesh(self) -> bool:
        """Flag indicating if the used mesh is real."""
        return self.beta is None

    @property
    def mesh(self) -> Union[MeshImFreq, MeshReFreq]:
        """The frequency mesh used for the calculation."""
        if self.is_real_mesh:
            mesh = MeshReFreq(*self.w_range, self.n_w)
        else:
            mesh = MeshImFreq(beta=self.beta, S="Fermion", n_max=self.n_iw)
        return mesh

    @property
    def n_cmpt(self) -> int:
        """The number of components."""
        conc = self.cast_cmpt()[0]
        return len(conc) if hasattr(conc, "__len__") else 1

    @property
    def spin_names(self) -> List[str]:
        return [s[0] for s in self.gf_struct]

    def resolve_location(self, root: Union[str, Path] = None) -> None:
        """Resolve the location path relative to the root directory.

        Parameters
        ----------
        root : Union[str, Path]
            The root directory of the calculation.
        """
        if root is None:
            root = "."
        root = Path(root)
        loc = Path(self.location)
        if not loc.is_absolute():
            loc = root / loc
        self.location = loc

    def _raw_dict(self) -> Dict[str, Any]:
        data = dict()
        for k, v in self.items():
            if v is not None:
                if isinstance(v, Parameters):
                    v = v.dict(include_none=False)
                data[k] = v
        return data

    def format_jobname(self, **kwargs) -> str:
        """Format the job name using the class attributes and given keyword arguments."""
        data = self._raw_dict()
        data.update(kwargs)
        return self._jobname.format(**data)

    def format_output(self, **kwargs) -> str:
        """Format the output file name using the class attributes and given keyword arguments."""
        data = self._raw_dict()
        data.update(kwargs)
        return self._output.format(**data)

    def format_location(self, **kwargs) -> str:
        """Format the location path using the class attributes and given keyword arguments."""
        data = self._raw_dict()
        data.update(kwargs)
        return self._location.format(**data)

    def set_mesh_re(self, w_min: float, w_max: float, n_w: int, eta: float = 0.0) -> None:
        """Set the mesh parameters for real frequencies.

        Parameters
        ----------
        w_min : float
            The minimum frequency.
        w_max : float
            The maximum frequency.
        n_w : int
            The number of mesh points.
        eta : float, optional
            The complex broadening.
        """
        self.n_w = n_w
        self.w_range = (w_min, w_max)
        self.eta = eta
        self.beta = None
        self.n_iw = None

    def set_mesh_im(self, n_iw: int, beta: float) -> None:
        """Set the mesh parameters for imaginary frequencies.

        Parameters
        ----------
        n_iw : int
            The number of Matsubara frequencies.
        beta : float
            The inverse temperature.
        """
        self.n_iw = n_iw
        self.beta = beta
        self.n_w = None
        self.w_range = None
        self.eta = None

    def add_solver(self, name: str, **kwargs) -> SolverParams:
        """Add a solver to the input parameters."""
        cls = SOLVERS.get(name, None)
        if cls is None:
            raise InputError(f"Solver '{name}' is not a registered solver!")

        self.solver_params = cls(**kwargs)
        return self.solver_params

    def add_maxent(self, **kwargs) -> MaxEntParams:
        """Add a MaxEnt solver to the input parameters."""
        self.maxent_params = MaxEntParams(**kwargs)
        return self.maxent_params

    def add_pade(self, **kwargs) -> PadeParams:
        """Add a Pade solver to the input parameters."""
        self.pade_params = PadeParams(**kwargs)
        return self.pade_params

    def dict(self, include_none: bool = True) -> Dict[str, Any]:
        """Return the input parameters as a dictionary."""
        data = super().dict(include_none)
        data["jobname"] = self.format_jobname()
        data["output"] = self.format_output()
        data["location"] = self.format_location()
        return data

    def validate(self) -> None:
        """Validate the input parameters."""
        # Check solver mesh compatibility
        if self.u:
            if self.solver_params is None:
                raise InputError("No solver set! Use `InputParameters.add_solver` to add a solver.")

            if self.solver_params.RE_MESH:
                # Check that n_w, w_range and eta are set
                if self.n_w is None:
                    raise InputMeshError("real", "n_w")
                if self.w_range is None:
                    raise InputMeshError("real", "w_range")
                # if self.eta == 0.0:
                #     raise InputMeshError("real", "eta")

            else:
                # Check that n_iw is set
                if self.n_iw is None:
                    raise InputMeshError("imaginary", "n_iw")

        # Check parameter shapes
        n_cmpt = len(self.conc) if hasattr(self.conc, "__len__") else 1
        if hasattr(self.u, "__len__"):
            assert len(self.u) == n_cmpt, "'u' does not match number of components!"
        if hasattr(self.eps, "__len__"):
            assert len(self.eps) == n_cmpt, "'eps' does not match number of components!"
        if hasattr(self.h_field, "__len__"):
            assert len(self.h_field) == n_cmpt, "'h' does not match number of components!"

        if self.symmetrize:
            h = self.h_field
            if isinstance(h, (int, float)):
                h = [h]
            if any(h):
                raise InputError("Cannot symmetrize with magnetic field!")

    def cast_cmpt(self) -> Tuple:
        """Cast the component parameters to numpy arrays."""
        conc = self.conc
        if isinstance(conc, (int, float)):
            conc = [conc]
        u = self.u
        eps = self.eps
        h = self.h_field
        if isinstance(u, (int, float)):
            u = [u] * len(conc)
        if isinstance(eps, (int, float)):
            eps = [eps] * len(conc)
        if isinstance(h, (int, float)):
            h = [h] * len(conc)
        assert sum(conc) == 1.0
        assert len(conc) == len(u) == len(eps) == len(h)
        return np.asarray(conc), np.asarray(u), np.asarray(eps)[:, None], np.asarray(h)[:, None]

    def loads(self, data: str) -> None:
        """Load input parameters from a TOML formatted string."""
        data = toml.loads(data)
        general = data["general"]
        solver = data.get("solver", dict())
        maxent = data.get("maxent", dict())
        pade = data.get("pade", dict())
        self.update(general)
        if solver:
            solver_name = solver.pop("type")
            self.add_solver(solver_name, **solver)
        if maxent:
            self.add_maxent(**maxent)
        if pade:
            self.add_pade(**pade)
        self.validate()

    def dumps(
        self,
        comments: bool = True,
        indent_values: bool = True,
        indent_comments: bool = True,
        linesep: Union[str, None] = "\n",
    ) -> str:
        """Dump the input parameters to a TOML formatted string."""
        if not linesep:
            linesep = os.linesep

        model_keys = [
            "lattice",
            "gf_struct",
            "half_bandwidth",
            "conc",
            "u",
            "eps",
            "h_field",
            "mu",
            "beta",
            "symmetrize",
        ]
        mesh_keys = [
            "n_iw",
            "n_w",
            "w_range",
            "eta",
        ]
        cpa_keys = [
            "method_cpa",
            "maxiter_cpa",
            "tol_cpa",
            "verbosity_cpa",
        ]
        dmft_keys = [
            "mixing_dmft",
            "gtol",
            "stol",
            "occ_tol",
        ]

        self.validate()
        data = self.dict(include_none=False)
        data.pop("jobname")
        data.pop("output")
        data.pop("location")
        solver_params = data.pop("solver_params", dict())
        maxent_params = data.pop("maxent_params", dict())
        pade_params = data.pop("pade_params", dict())

        # General section
        general = toml.table()
        general.add(toml.nl())
        general.add("jobname", self._jobname)
        general.add("location", self._location)
        general.add("tmpdir", data.pop("tmpdir"))
        general.add("output", self._output)
        general.add("n_loops", data.pop("n_loops"))
        general.add("load_iter", data.pop("load_iter"))

        # general.update(data)
        items = {k: self.get(k) for k in model_keys if self.get(k) is not None}
        general.add(toml.nl())
        general.add(toml.comment("Model parameters"))
        general.update(items)

        items = {k: self.get(k) for k in mesh_keys if self.get(k) is not None}
        if items:
            general.add(toml.nl())
            general.add(toml.comment("Mesh parameters"))
            general.update(items)

        items = {k: self.get(k) for k in cpa_keys if self.get(k) is not None}
        if items:
            general.add(toml.nl())
            general.add(toml.comment("CPA parameters"))
            general.update(items)

        items = {k: self.get(k) for k in dmft_keys if self.get(k) is not None}
        if items:
            general.add(toml.nl())
            general.add(toml.comment("DMFT parameters"))
            general.update(items)

        # Solver section
        if solver_params:
            solver = toml.table()
            solver.add(toml.nl())
            solver_type = solver_params.pop("type")
            solver.add("type", solver_type)
            solver.update(solver_params)
        else:
            solver = None

        # maxent section
        if maxent_params:
            maxent = toml.table()
            maxent.add(toml.nl())
            maxent.update(maxent_params)
        else:
            maxent = None

        # pade section
        if pade_params:
            pade = toml.table()
            pade.add(toml.nl())
            pade.update(pade_params)
        else:
            pade = None

        # Format document
        doc = toml.document()
        frmt = "%Y-%m-%d"
        doc.add(toml.comment(f"Generated on {datetime.now().strftime(frmt)}"))
        # doc.add(toml.nl())
        # doc.add(toml_section_comment("General parameters that apply to the whole simulation."))
        doc.add(toml.nl())
        doc.add("general", general)
        # doc.add(toml.nl())
        # doc.add(toml_section_comment("Solver specific parameters"))
        if solver is not None:
            doc.add(toml.nl())
            doc.add("solver", solver)
        if maxent is not None:
            doc.add(toml.nl())
            doc.add("maxent", maxent)
        if pade is not None:
            doc.add(toml.nl())
            doc.add("pade", pade)
        # doc.add(toml.nl())

        # Add comments
        if comments:
            for key in general.keys():
                try:
                    general[key].comment(self.__descriptions__[key])
                except (AttributeError, KeyError):
                    pass
            if solver:
                for key in solver.keys():
                    try:
                        solver[key].comment(self.solver_params.__descriptions__[key])
                    except (AttributeError, KeyError):
                        pass
                    solver["type"].comment("Solver used to solve the impurity problem.")
            if maxent:
                for key in maxent.keys():
                    try:
                        maxent[key].comment(self.maxent_params.__descriptions__[key])
                    except (AttributeError, KeyError):
                        pass

            if pade:
                for key in pade.keys():
                    try:
                        pade[key].comment(self.pade_params.__descriptions__[key])
                    except (AttributeError, KeyError):
                        pass

        text = toml.dumps(doc)
        lines = text.splitlines(keepends=False)
        if indent_values:
            _indent_values(lines)
        if indent_comments:
            _indent_comments(lines)
        return linesep.join(lines)

    def load(self, file: Union[str, Path] = None, missing_ok: bool = False) -> "InputParameters":
        """Load input parameters from a TOML formatted file."""
        file = Path(file) if file is not None else self._path
        if file is None:
            return self
        if not file.exists():
            if missing_ok:
                print(f"File not found: {file}")
                return self
            raise FileNotFoundError(f"File not found: {file}")
        self.loads(file.read_text())

        if self._path is not None:
            self.resolve_location(self._path.parent)

        return self

    def dump(
        self, file: Union[str, Path] = None, mkdir: bool = False, **kwargs
    ) -> "InputParameters":
        """Dump the input parameters to a TOML formatted file."""
        file = Path(file) if file is not None else self._path
        if mkdir:
            file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(self.dumps(**kwargs))
        return self

    def copy_file(self, file: Union[str, Path]) -> "InputParameters":
        """Copy the input parameters to a new file."""
        file = Path(file)
        file.write_text(self.dumps())
        return InputParameters(file)

    @classmethod
    def __factory_from_dict__(cls, name: str, data: Dict[str, Any]) -> "InputParameters":
        if "gf_struct" in data:
            data["gf_struct"] = list(list(x) for x in data.pop("gf_struct"))
        for key, val in data.items():
            if isinstance(val, tuple):
                data[key] = list(val)
        solver = data.pop("solver_params", dict())
        for key, val in solver.items():
            if isinstance(val, tuple):
                solver[key] = list(val)

        maxent = data.pop("maxent_params", dict())
        for key, val in maxent.items():
            if isinstance(val, tuple):
                maxent[key] = list(val)

        pade = data.pop("pade_params", dict())
        for key, val in pade.items():
            if isinstance(val, tuple):
                pade[key] = list(val)

        self = cls()
        self.update(data)
        if solver:
            solver_name = solver.pop("type")
            self.add_solver(solver_name, **solver)
        if maxent:
            self.add_maxent(**maxent)
        if pade:
            self.add_pade(**pade)
        return self

    def __reduce_to_dict__(self) -> Dict[str, Any]:
        data = self.dict(include_none=False)
        data["gf_struct"] = tuple(tuple(x) for x in data.pop("gf_struct"))
        for key, val in data.items():
            if isinstance(val, list):
                data[key] = tuple(val)
        solver = data.get("solver_params", dict())
        for key, val in solver.items():
            if isinstance(val, list):
                solver[key] = tuple(val)
        maxent = data.get("maxent_params", dict())
        for key, val in maxent.items():
            if isinstance(val, list):
                maxent[key] = tuple(val)
        pade = data.get("pade_params", dict())
        for key, val in pade.items():
            if isinstance(val, list):
                pade[key] = tuple(val)
        return data

    def flatten(self, include_none: bool = True) -> Dict[str, Any]:
        """Flatten the input parameters to a single layer dictionary."""
        data = self.dict(include_none=include_none)
        for key, val in data.pop("solver_params", dict()).items():
            data[f"solver.{key}"] = val
        return data

    def __getitem__(self, key):
        if key.startswith("solver."):
            return self.solver_params[key.replace("solver.", "")]
        elif key.startswith("maxent."):
            return self.maxent_params[key.replace("maxent.", "")]
        elif key.startswith("pade."):
            return self.pade_params[key.replace("pade.", "")]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key.startswith("solver."):
            self.solver_params[key.replace("solver.", "")] = value
        elif key.startswith("maxent."):
            self.maxent_params[key.replace("maxent.", "")] = value
        elif key.startswith("pade."):
            self.pade_params[key.replace("pade.", "")] = value
        else:
            super().__setitem__(key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.format_jobname()})"


# Register classes for serialization with h5
register_class(Parameters)
register_class(FtpsSolverParams)
register_class(CthybSolverParams)
register_class(MaxEntParams)
register_class(PadeParams)
register_class(InputParameters)
register_class(HubbardISolverParams)

# Register solver input classes
register_solver_input(FtpsSolverParams)
register_solver_input(CthybSolverParams)
register_solver_input(HubbardISolverParams)
register_solver_input(HartreeSolverParams)
