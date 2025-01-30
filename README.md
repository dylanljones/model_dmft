# Model DMFT

A versatile python wrapper to perform CPA+DMFT calculations utilizing the TRIQS software library.

> [!WARNING]
>
> This project is still under development and is not yet ready for production use.
> It might contain critical bugs and the API is subject to change.


## Installation

Install the library by running the following command:

```bash
pip install "git+ssh://git@git.rz.uni-augsburg.de/jonesdyl/model_dmft"
```

### Dependencies

The following dependencies are required to run the code and can not be installed via pip:

- [TRIQS]
- [CTHYB]
- [ForkTPS]

It is recommended to install [TRIQS] via conda in a separate environment.
First, create a new environment and activate it:
```bash
conda create -n triqs python=3.10 -y && conda activate triqs
```

Then, we can install the [TRIQS] framework:
```bash
conda install -c conda-forge triqs=3.1.1 -y
```

The [ForkTPS] solver isn't public yet, so it can't be installed via conda.
Follow the instructions in the (private) [ForkTPS] repository to install it.


The [CTHYB] solver, however, can be installed via conda:
```bash
conda install -c conda-forge triqs_cthyb -y
```

TL;DR:
```bash
conda create -n triqs python=3.12 -y
conda activate triqs && conda install -c conda-forge triqs=3.1.1 -y
conda activate triqs && conda install -c conda-forge triqs_cthyb -y
```

## Usage

CPA-DMFT calculation can be run via a script or the command line interface.
For running a calculation via a script, initialize the input parameter object and
call the `solve` method:

```python
from model_dmft import InputParameters
from model_dmft.runner import solve

params = InputParameters(solver="ftps")
params.restart = True
params.location = "directory"
params.set_mesh_re(-6, 6, 1000, eta=0.02)

params.u = 1.0
params.eps = [-0.5, 0.5]
params.conc = [0.2, 0.8]

solve(params, n_procs=4)
```

See the input file section for more information on the input parameters.


For running a calculation via the command line interface, you need to create an input file `inp.toml`
and run the following command:
```bash
model_dmft run inp.toml -n 4
```
The `-n` flag specifies the number of MPI processes (`n_procs`) to use **for each component solver**.
The solvers run in parallel for each component, so the total number of MPI processes is the
number of components times the number of MPI processes specified.


### Input file

The input file uses the [TOML](https://toml.io/en/) file format. It contains two sections: `[general]` and `[solver]`.

The `[general]` section contains general parameters that apply to the whole simulation,
including metadata like the job name, email address, and output file name and general model parameters
like the lattice type, Green's function structure, and computational parameters like the number of DMFT loops.

The `[solver]` section specifies wich solver is used for the impurity problem and
contains solver-specific parameters.

A full example of an input file and SLURM script can be found in the [example](example) directory.
To start a new calculation, copy the folder and modify the parameters in the `inp.toml` file as needed:
```shell
cp <path-to-repo>/example <new-directory>
```

#### ``[general]``
| Name             | Type                     | Description                                                                 |
|------------------|--------------------------|-----------------------------------------------------------------------------|
| `jobname`        | `str`                    | Name of the job.                                                            |
| `location`       | `str`                    | Location of the job.                                                        |
| `output`         | `str`                    | Name of the output file.                                                    |
| `tmp_dir`        | `str`                    | Temporary directory.                                                        |
| `n_loops`        | `int`                    | Number of DMFT loops.                                                       |
| `restart`        | `bool`                   | Restart from previous calculation.                                          |
| `lattice`        | `str`                    | Lattice type.                                                               |
| `gf_struct`      | `List[List[str, int]]`   | Green's function structure.                                                 |
| `half_bandwidth` | `float`                  | Half bandwidth.                                                             |
| `t`              | `float`                  | Hopping parameter, can be used *instead* of `half_bandwidth`.               |
| `conc`           | `float` or `List[float]` | Concentrations of components.                                               |
| `eps`            | `float` or `List[float]` | Local energy levels of components.                                          |
| `u`              | `float` or `List[float]` | Coulomb interaction of components.                                          |
| `h_field`        | `float` or `List[float]` | Magnetic field.                                                             |
| `mu`             | `float`                  | Chemical potential.                                                         |
| `beta`           | `float`                  | Inverse temperature. If `beta=0` a real frequency calculation is performed. |
| `n_w`            | `int`                    | Number of real frequency points.                                            |
| `w_range`        | `List[float]`            | Real frequency range.                                                       |
| `eta`            | `float`                  | Imaginary broadening used for real frequency calculation.                   |
| `n_iw`           | `int`                    | Number of Matsubara frequency points.                                       |
| `method_cpa`     | `str`                    | CPA method.                                                                 |
| `maxiter_cpa`    | `int`                    | Maximum number of iterations for CPA (if `method_cpa="iter"`).              |
| `verbosity_cpa`  | `int`                    | Verbosity level of CPA iterations.                                          |
| `tol_cpa`        | `float`                  | Tolerance for CPA.                                                          |
| `gtol`           | `float`                  | Convergence tolerance for the coherent Green's function.                    |
| `stol`           | `float`                  | Convergence tolerance for the self energy.                                  |
| `occ_tol`        | `float`                  | Convergence tolerance for occupation number.                                |
| `mixing_cpa`     | `float`                  | Mixing parameter for CPA.                                                   |
| `mixing_dmft`    | `float`                  | Mixing parameter for DMFT.                                                  |


```toml
[general]  # General parameters that apply to the whole simulation.
jobname         = "FTPS-CPA+DMFT"        # The job name of the simulation. Used for generating SLURM scripts.
location        = "."                    # The directory where the simulation is run.
output          = "out.h5"               # The name of the output file.
tmp_dir         = ".tmp/"                # The directory where temporary files are stored.
n_loops         = 3                      # The total number of iterations to perfom.
restart         = true                   # Flag if the calculation should resume from previous results or start over.

# Model parameters
lattice         = "bethe"                # The lattice type
gf_struct       = [["up", 1], ["dn", 1]] # The structure of the Greens function
# half_bandwidth = 2.0                   # The half bandwidth of the lattice DOS
t               = 1.0                    # The hopping parameter (alternative to half_bandwidth)
conc            = [0.7, 0.3]             # The concentrations of the components.
eps             = [-1.0, +1.0]           # The on-site energies of the components.
h_field         = 0                      # The magnetic Zeeman field of the components.
u               = 4                      # The Hubbard interaction strength (Coulomb repulsion) of the components.
mu              = 0                      # The chemiocal potential
# beta          = 10                     # The inverse temperature

# REAL MESH (if beta is not given)
n_w             = 3001                   # Number of mesh points
w_range         = [-8, 8]                # Frequency range
eta             = -1                     # Complex broadening (Negative: eta = 6 / nbath)
# MATSUBARA MESH (if beta given)
# n_iw          = 1028                   # Number of mesh points

# CPA parameters
method_cpa     = "iter"                  # Method used for computing the CPA self energy.
maxiter_cpa    = 10000                   # Number of iterations for computing the CPA self energy.
tol_cpa        = 1e-06                   # Tolerance for the CPA self-consistency check.
verbosity_cpa  = 1                       # Verbosity level of the CPA iterations.

# DMFT parameters
mixing_dmft    = 0.1                      # Mixing of the DMFT self energy.

# Convergence parameters
gtol           = 1e-06                    # Convergence tolerance for the coherent Green's function.
stol           = 1e-06                    # Convergence tolerance for the self energy.
occ_tol        = 1e-06                    # Occupation tolerance for CPA.
```

#### ``[solver]``

| Name   | Type               | Description                               |
|--------|--------------------|-------------------------------------------|
| `type` | `{'ftps', 'cthyb}` | The solver used for the impurity problem. |
| ...    | ...                | Solver-specific parameters.               |


#### `ftps`

Parameters for the [ForkTPS] solver.

| Name          | Type    | Description                                            |
|---------------|---------|--------------------------------------------------------|
| `bath_fit`    | `bool`  | DiscretizeBath vs BathFitter                           |
| `n_bath`      | `int`   | Number of bath sites.                                  |
| `time_steps`  | `int`   | Number of time steps for the time evolution.           |
| `dt`          | `float` | Time step for the time evolution.                      |
| `method`      | `str`   | Time evolution method, "TDVP", "TDVP_2" or "TEBD"      |
| `sweeps`      | `int`   | Number of DMRG sweeps.                                 |
| `tw`          | `float` | Truncated weight for every link. (default 1e-9)        |
| `maxm`        | `int`   | Maximum bond dimension. (default 100)                  |
| `nmax`        | `int`   | Maximal Number of Krylov vectors created. (default 40) |

```toml
[solver]
type            = "ftps"                 # Solver used to solve impurity problem

bath_fit        = true                   # DiscretizeBath vs BathFitter
n_bath          = 20                     # The number of bath sites used by the FTPS solver
# Tevo parameters
time_steps      = 100                    # Number of time steps for the time evolution.
dt              = 0.1                    # Time step for the time evolution.
method          = "TDVP_2"               # Time evolution method, "TDVP", "TDVP_2" or "TEBD"
# DMRG parameters
sweeps          = 10                     # Number of DMRG sweeps (default 15)
# Used for Tevo and DMRG parameters
tw              = 1e-9                   # Truncated weight for every link. (default 1e-9)
maxm            = 100                    # Maximum bond dimension. (default 100)
nmax            = 40                     # Maximal Number of Krylov vectors created. (default 40)
```


#### `cthyb`

Parameters for the [CTHYB] solver (experimental).

| Name               | Type    | Description                                  |
|--------------------|---------|----------------------------------------------|
| `n_cycles`         | `int`   | Number of Quantum Monte Carlo cycles.        |
| `n_warmup_cycle`   | `int`   | Number of warmup cycles.                     |
| `length_cycle`     | `int`   | Length of the cycle.                         |
| `n_tau`            | `int`   | Number of imaginary time steps.              |
| `perform_tail_fit` | `bool`  | Perform tail fit of Sigma and G.             |
| `fit_max_moment`   | `int`   | Highest moment to fit in the tail of Sigma.  |
| `fit_min_n`        | `int`   | Index of iw from which to start fitting.     |
| `fit_max_n`        | `int`   | Index of iw up to which to fit.              |


```toml
[solver]
type             = "cthyb"               # Solver used to solve impurity problem

n_cycles         = 5000                  # Number of Quantum Monte Carlo cycles (default 10_000)
n_warmup_cycle   = 1000                  # Number of warmup cycles (default: 1_000)
length_cycle     = 100                   # Length of the cycle (default: 100)
n_tau            = 10001                 # Number of imaginary time steps. (default: 10001)
perform_tail_fit = false                 # Perform tail fit of Sigma and G (default: false)
fit_max_moment   = 3                     # Highest moment to fit in the tail of Sigma (default: 3)
fit_min_n        = 0                     # Index of iw from which to start fitting (default: 0.5*n_iw)
fit_max_n        = 0                     # Index of iw up to which to fit (default: n_iw)
```

[TRIQS]: https://triqs.github.io/triqs/latest/index.html
[CTHYB]: https://triqs.github.io/cthyb/latest/index.html
[ForkTPS]: https://git.rz.uni-augsburg.de/itphy-sw-origin/forktps
