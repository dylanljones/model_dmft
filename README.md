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

Additionally, one or multiple of the following solvers are required:

- [CTHYB]
- [ForkTPS]
- [HubbardI]
- [Hartree-Fock]

It is recommended to install [TRIQS] via conda in a separate environment.
First, create a new environment and activate it:
```bash
conda create -n triqs python=3.10 -y && conda activate triqs && conda install -c conda-forge triqs -y
```

See the [INSTALLATION.md](INSTALLATION.md) file for more detailed installation instructions including the solvers.

## Usage

CPA-DMFT calculation can be run via a script or the command line interface.
For running a calculation via a script, initialize the input parameter object and
call the `solve` method:

```python
from model_dmft import InputParameters
from model_dmft.runner import solve

params = InputParameters(solver="ftps")
params.load_iter = -1
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

The input file uses the [TOML](https://toml.io/en/) file format. It contains two main sections: `[general]` and `[solver]`.

The `[general]` section contains general parameters that apply to the whole simulation,
including metadata like the job name, email address, and output file name and general model parameters
like the lattice type, Green's function structure, and computational parameters like the number of DMFT loops.

The `[solver]` section specifies wich solver is used for the impurity problem and
contains solver-specific parameters.

For post-processing of imaginary mesh calculations on Matsubara frequencies, there are two additional sections:
`[maxent]` or `[pade]`.


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
| `tmpdir`         | `str`                    | Temporary directory.                                                        |
| `n_loops`        | `int`                    | Number of DMFT loops.                                                       |
| `load_iter`      | `int`                    | Continue from specific iteration (-1 for last iteration, 0 for restart).    |
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
| `symmetrize`     | `bool`                   | Symmetrize the spin channels (if no field).                                 |
| `n_w`            | `int`                    | Number of real frequency points (if beta not set).                          |
| `w_range`        | `List[float]`            | Real frequency range (if beta not set).                                     |
| `eta`            | `float`                  | Imaginary broadening used for real frequency calculation (if beta not set). |
| `n_iw`           | `int`                    | Number of Matsubara frequency points (if beta set).                         |
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
load_iter       = -1                     # Continue from specific iteration (-1 for last iteration, 0 for restart).

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
symmetrize      = false                  # Symmetrize the spin channels (if no field)

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
mixing_dmft    = 0.1                     # Mixing of the DMFT self energy.

# Convergence parameters
gtol           = 1e-06                   # Convergence tolerance for the coherent Green's function.
stol           = 1e-06                   # Convergence tolerance for the self energy.
occ_tol        = 1e-06                   # Occupation tolerance for CPA.
```

#### ``[solver]``

| Name   | Type                                       | Description                               |
|--------|--------------------------------------------|-------------------------------------------|
| `type` | `{'ftps', 'cthyb', 'hubbardI', 'hartree'}` | The solver used for the impurity problem. |
| ...    | ...                                        | Solver-specific parameters.               |


#### `ftps`

Parameters for the [ForkTPS] solver.

| Name            | Type           | Description                                                                             |
|-----------------|----------------|-----------------------------------------------------------------------------------------|
| `bath_fit`      | `bool`         | DiscretizeBath vs BathFitter                                                            |
| `n_bath`        | `int`          | Number of bath sites                                                                    |
| `enforce_gap`   | `List[float]`  | Enforce a gap in the bath spectrum (default: None)                                      |
| `ignore_weight` | `float`        | Ignore bath states with weight below this threshold (default: 0.0)                      |
| `sweeps`        | `int`          | Number of DMRG sweeps (default: 15)                                                     |
| `dmrg_maxm`     | `int`          | Maximum bond dimension for DMRG, if not specified `maxm` (default: None)                |
| `dmrg_maxmI`    | `int`          | Maximum imp-imp bond dimension for DMRG, if not specified `maxmI` (default: None)       |
| `dmrg_maxmIB`   | `int`          | Maximum imp-bath bond dimension for DMRG, if not specified `maxmIB` (default: None)     |
| `dmrg_maxmB`    | `int`          | Maximum bath-bath bond dimension for DMRG, if not specified `maxmB` (default: None)     |
| `dmrg_nmax`     | `int`          | Maximal Number of Krylov vectors for DMRG, if not specified `nmax` (default: None)      |
| `dmrg_tw`       | `float`        | Truncated weight for DMRG, if not specified `tw` (default: None)                        |
| `dt`            | `float`        | Time step for the time evolution (default: 0.1)                                         |
| `time_steps`    | `int`          | Number of time steps for the time evolution, optimized if not specified (default: None) |
| `method`        | `str`          | Time evolution method, "TDVP", "TDVP_2" or "TEBD" (default: "TDVP_2")                   |
| `maxm`          | `int`          | Maximum bond dimension. (default 100)                                                   |
| `maxmI`         | `int`          | Maximum imp-imp bond dimension. (default 100)                                           |
| `maxmIB`        | `int`          | Maximum imp-bath bond dimension. (default 100)                                          |
| `maxmB`         | `int`          | Maximum bath-bath bond dimension. (default 100)                                         |
| `nmax`          | `int`          | Maximal Number of Krylov vectors created. (default 40)                                  |
| `tw`            | `float`        | Truncated weight for every link. (default 1e-9)                                         |

```toml
[solver]
type            = "ftps"                 # Solver used to solve impurity problem

bath_fit        = true                   # DiscretizeBath vs BathFitter
n_bath          = 20                     # The number of bath sites used by the FTPS solver
time_steps      = 100                    # Number of time steps for the time evolution.
dt              = 0.1                    # Time step for the time evolution.
method          = "TDVP_2"               # Time evolution method, "TDVP", "TDVP_2" or "TEBD"
sweeps          = 10                     # Number of DMRG sweeps (default 15)
tw              = 1e-9                   # Truncated weight for every link. (default 1e-9)
maxm            = 100                    # Maximum bond dimension. (default 100)
nmax            = 40                     # Maximal Number of Krylov vectors created. (default 40)
```


#### `cthyb`

Parameters for the [CTHYB] solver.

| Name             | Type    | Description                                                                  |
|------------------|---------|------------------------------------------------------------------------------|
| `n_cycles`       | `int`   | Number of Quantum Monte Carlo cycles (default: 10_000)                       |
| `n_warmup_cycle` | `int`   | Number of warmup cycles (default: 1_000)                                     |
| `length_cycle`   | `int`   | Length of the cycle (default: 100)                                           |
| `n_tau`          | `int`   | Number of imaginary time steps (default: 10_001)                             |
| `density_matrix` | `bool`  | Measure the impurity density matrix (default: False)                         |
| `measure_g_l`    | `bool`  | Measure G_l (Legendre) (default: False)                                      |
| `n_l`            | `int`   | Number of Legendre polynomials. (default: 30)                                |
| `legendre_fit`   | `bool`  | Fit Green's function and self energy using Legendre Gf (default: false)      |
| `tail_fit `      | `bool`  | Perform tail fit of Sigma and G (default: False)                             |
| `fit_max_moment` | `int`   | Highest moment to fit in the tail of Sigma (default: 3)                      |
| `fit_min_n`      | `int`   | Index of iw from which to start fitting (default: 0.8*n_iw)                  |
| `fit_max_n`      | `int`   | Index of iw up to which to fit (default: n_iw)                               |
| `fit_min_w`      | `float` | iw from which to start fitting (default: None)                               |
| `fit_max_w`      | `float` | iw up to which to fit (default: None)                                        |
| `crm_dyson`      | `bool`  | Solve Dyson equation using constrained minimization problem (default: false) |
| `crm_wmax`       | `float` | Spectral width of the impurity problem for DLR basis (default: None)         |
| `crm_eps`        | `float` | Accuracy of the DLR basis to represent Green’s function (default: 1e-8)      |

```toml
[solver]
type            = "cthyb"                # Solver used to solve impurity problem

n_cycles        = 5000                   # Number of Quantum Monte Carlo cycles (default 10_000)
n_warmup_cycle  = 1000                   # Number of warmup cycles (default: 1_000)
length_cycle    = 100                    # Length of the cycle (default: 100)
n_tau           = 10001                  # Number of imaginary time steps. (default: 10001)
tail_fit        = false                  # Perform tail fit of Sigma and G (default: false)
fit_max_moment  = 3                      # Highest moment to fit in the tail of Sigma (default: 3)
fit_min_n       = 0                      # Index of iw from which to start fitting (default: 0.5*n_iw)
fit_max_n       = 0                      # Index of iw up to which to fit (default: n_iw)
measure_g_l     = false                  # Measure G_l (Legendre) (default: false)
n_l             = 30                     # Number of Legendre polynomials (default: 30)
density_matrix  = true                   # Measure the impurity density matrix (default: false)
```


#### `hubbardI`

Parameters for the [HubbardI] solver (experimental).

| Name             | Type    | Description                                                                  |
|------------------|---------|------------------------------------------------------------------------------|
| `n_tau`          | `int`   | Number of imaginary time steps (default: 10_001)                             |
| `density_matrix` | `bool`  | Measure the impurity density matrix (default: False)                         |
| `measure_g_l`    | `bool`  | Measure G_l (Legendre) (default: False)                                      |
| `measure_g_tau`  | `bool`  | Measure G_tau (default: False)                                               |
| `legendre_fit`   | `bool`  | Fit Green's function and self energy using Legendre Gf (default: false)      |
| `n_l`            | `int`   | Number of Legendre polynomials. (default: 30)                                |

```toml
[solver]
type            = "hubbardI"             # Solver used to solve impurity problem

n_tau           = 10001                  # Number of imaginary time steps. (default: 10001)
measure_g_l     = true                   # Measure G_l (Legendre) (default: false)
n_l             = 30                     # Number of Legendre polynomials (default: 30)
density_matrix  = true                   # Measure the impurity density matrix (default: false)
legendre_fit    = true                   # Fit Green's function and self energy using Legendre Gf (default: false)
```


#### `hartree`

Parameters for the [Hartree-Fock] solver (experimental).

| Name         | Type    | Description                                                         |
|--------------|---------|---------------------------------------------------------------------|
| `one_shot`   | `bool`  | Perform a one-shot or self-consitent root finding (default: False)  |
| `method`     | `str`   | Method used for root finding (default: 'krylov')                    |
| `tol`        | `float` | Tolerance for root finder if one_shot=False (default: 1e-5)         |
| `with_fock`  | `bool`  | Include Fock exchange terms in the self-energy (default: False)     |
| `force_real` | `bool`  | Force the self energy from Hartree fock to be real (default: True)  |

```toml
[solver]
type            = "hartree"              # Solver used to solve impurity problem

one_shot        = false                  # Perform a one-shot or self-consitent root finding (default: False)
method          = "krylov"               # Method used for root finding (default: 'krylov')
tol             = 1e-5                   # Tolerance for root finder if one_shot=False (default: 1e-5)
with_fock       = false                  # Include Fock exchange terms in the self-energy (default: False)
force_real      = true                   # Force the self energy from Hartree fock to be real (default: True)
```


#### `pade`

Parameters for the Pade analytical continuation.

| Name            | Type          | Description                                  |
|-----------------|---------------|----------------------------------------------|
| `n_w`           | `int`         | Number of real frequency points.             |
| `w_range`       | `List[float]` | Real frequency range.                        |
| `n_points`      | `int`         | Number of points for the Pade approximation. |
| `freq_offset`   | `int`         | Frequency offset for the Pade approximation. |

```toml
[pade]

n_w             = 2001                   # Number of real frequency points
w_range         = [-6, 6]                # Range of real frequencies
n_points        = 100                    # Number of points for the Pade approximation
freq_offset     = 1e-3                   # Frequency offset for the Pade approximation
```


#### `maxent`

Parameters for the Maximum Entropy analytical continuation.

| Name              | Type          | Description                                  |
|-------------------|---------------|----------------------------------------------|
| `error`           | `int`         | Error threshold (default: 1e-4).             |
| `cost_function`   | `List[float]` | Cost function (default: bryan).              |
| `probability`     | `int`         | Probability distribution (default: normal)   |
| `mesh_type_alpha` | `int`         | Alpha mesh type (default: logarithmic).      |
| `n_alpha`         | `int`         | Number of alpha mesh points (default: 60).   |
| `alpha_range`     | `int`         | Alpha range (default: [0.01, 2000]).         |
| `mesh_type_w`     | `int`         | Frequency mesh type (default: hyperbolic).   |
| `n_w`             | `int`         | Number of real frequency points.             |
| `w_range`         | `List[float]` | Real frequency range.                        |


```toml
[maxent]

error           = 1e-4                   # Error threshold
cost_function   = [0.0, 0.0, 0.0]        # Cost function
probability     = "normal"               # Probability distribution
mesh_type_alpha = "logarithmic"          # Alpha mesh type
n_alpha         = 60                     # Number of alpha mesh points
alpha_range     = [0.01, 2000]           # Alpha range
mesh_type_w     = "hyperbolic"           # Frequency mesh type
n_w             = 2001                   # Number of real frequency points
w_range         = [-6, 6]                # Range of real frequencies
```


[TRIQS]: https://triqs.github.io/triqs/latest/index.html
[CTHYB]: https://github.com/TRIQS/cthyb
[HubbardI]: https://github.com/TRIQS/hubbardI
[Hartree-Fock]: https://github.com/TRIQS/hartree_fock
[ForkTPS]: https://git.rz.uni-augsburg.de/itphy-sw-origin/forktps
