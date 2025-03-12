# Installation


Install the library by running the following command:

```bash
pip install "git+ssh://git@git.rz.uni-augsburg.de/jonesdyl/model_dmft"
```

The following dependencies are required to run the code and can not be installed via pip:

- [TRIQS]

Additionally, one or multiple of the following solvers are required:

- [CTHYB]
- [HubbardI]
- [Hartree-Fock]
- [ForkTPS]


## TRIQS

It is recommended to install [TRIQS] via conda in a separate environment.
First, create a new environment and activate it:
```bash
conda create -n triqs python=3.10 -y && conda activate triqs
```

Then, install TRIQS:
```bash
conda install -c conda-forge triqs -y
```


## Hartree-Fock

1. Clone the latest stable version of the [Hartree-Fock] repository:
   ```bash
   git clone https://github.com/TRIQS/hartree_fock hartree_fock.src
   ```

2. Create and move to a new directory where you will compile the code:
   ```bash
   mkdir hartree_fock.build && cd hartree_fock.build
   ```

3. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation:
   ```bash
   source path_to_triqs/share/triqs/triqsvars.sh
   ```
   If you are using TRIQS from Anaconda, you can use the ``CONDA_PREFIX`` environment variable:
   ```bash
   source $CONDA_PREFIX/share/triqs/triqsvars.sh
   ```

4. In the build directory call cmake, including any additional custom CMake options, see below:
   ```bash
   cmake ../hartree_fock.src
   ```

5. Finally, compile the code and install the application:
   ```bash
   make install
   ```


## HubbardI

1. Clone the latest stable version of the [CTHYB] repository:
   ```bash
   git clone https://github.com/TRIQS/hubbardI hubbardI.src
   ```

2. Create and move to a new directory where you will compile the code:
   ```bash
   mkdir hubbardI.build && cd hubbardI.build
   ```

3. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation:
   ```bash
   source path_to_triqs/share/triqs/triqsvars.sh
   ```
   If you are using TRIQS from Anaconda, you can use the ``CONDA_PREFIX`` environment variable:
   ```bash
   source $CONDA_PREFIX/share/triqs/triqsvars.sh
   ```

4. In the build directory call cmake, including any additional custom CMake options, see below:
   ```bash
   cmake ../hubbardI.src
   ```

5. Finally, compile the code and install the application:
   ```bash
   make
   make install
   ```


## CTHYB

1. Clone the latest stable version of the [CTHYB] repository:
   ```bash
   git clone https://github.com/TRIQS/cthyb cthyb.src
   ```

2. Create and move to a new directory where you will compile the code:
   ```bash
   mkdir cthyb.build && cd cthyb.build
   ```

3. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation:
   ```bash
   source path_to_triqs/share/triqs/triqsvars.sh
   ```
   If you are using TRIQS from Anaconda, you can use the ``CONDA_PREFIX`` environment variable:
   ```bash
   source $CONDA_PREFIX/share/triqs/triqsvars.sh
   ```

4. In the build directory call cmake, including any additional custom CMake options, see below:
   ```bash
   cmake ../cthyb.src
   ```

5. Finally, compile the code and install the application:
   ```bash
   make
   make install
   ```


## ForkTPS

The [ForkTPS] solver is not public yet!
The repository is not available on GitHub and must be cloned from the private
University of Augsburg GitLab repository.

> [!IMPORTANT]
>
> The current version of *forktps* is only compatible with TRIQS v3.1.x.
> The latest supported version is TRIQS v3.1.1.

Make sure the right version of TRIQS is installed before proceeding with the installation of *forktps*:

```bash
conda create -n triqs_ftps python=3.10 -y && conda activate triqs_ftps && conda install -c conda-forge triqs=3.1.1 -y
```

### ITensor

[ForkTPS] requires the [ITensor] library to be installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/ITensor/ITensor
   ```
2. Edit the configuration file by making a copy of the template:
   ```bash
   cd ITensor
   cp options.mk.sample options.mk
   ```
3. Finally, build the library by running:
   ```bash
   make
   ```
4. Set the environment variable ```export ITENSOR_ROOT=/Path/To/ITensor``` to the root
   directory of the ITensor installation or add the following line to your ```.bashrc```:
   ```bash
   export ITENSOR_ROOT=/Path/To/ITensor
   ```

### Installing ForkTPS

Once [ITensor] is installed, you can proceed with the installation of [ForkTPS]:

1. Clone the latest stable version of the [ForkTPS] repository:
   ```bash
   git clone https://git.rz.uni-augsburg.de/itphy-sw-origin/forktps forktps.src
   ```

2. Create and move to a new directory where you will compile the code:
   ```bash
   mkdir forktps.build && cd forktps.build
   ```

3. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation:
   ```bash
   source path_to_triqs/share/triqs/triqsvars.sh
   ```
   If you are using TRIQS from Anaconda, you can use the ``CONDA_PREFIX`` environment variable:
   ```bash
   source $CONDA_PREFIX/share/triqs/triqsvars.sh
   ```

4. In the build directory call cmake, including any additional custom CMake options, see below:
   ```bash
   cmake ../forktps.src -DBuild_Tests=OFF
   ```

5. Finally, compile the code and install the application:
   ```bash
   make
   make install
   ```


### Troubleshooting

Some common issues and their solutions are listed below:

- `crypt.h: No such file or directory`

    You need to install the `libxcrypt` package and export the include path:
    ```bash
    conda install --channel=conda-forge libxcrypt
    export CPATH=/opt/conda/include/
    ```

- `libcurl.so.4: no version information available (required by cmake)`

  You need to install/upgrade the `libcurl` and `cmake` package:

  ```bash
  conda install curl cmake -y
  ```
  or
  ```bash
  conda upgrade curl cmake
  ```


- `libtinfo.so.6: no version information available (required by /bin/sh)`

  You need to install the `ncurses` package:
  ```bash
  conda install -c conda-forge ncurses
  ```


- `/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer`

  You need to downgrade the `libffi` package, for example:
  ```bash
  conda install libffi==3.3
  ```

- If you are having problems compiling the code with tests (`gtest`), you can disable them by
  setting the cmake flag `-DBuild_Tests=OFF`:

  ```bash
  cmake ../forktps.src -DBuild_Tests=OFF
  ```


[TRIQS]: https://triqs.github.io/triqs/latest/index.html
[CTHYB]: https://github.com/TRIQS/cthyb
[HubbardI]: https://github.com/TRIQS/hubbardI
[Hartree-Fock]: https://github.com/TRIQS/hartree_fock
[ForkTPS]: https://git.rz.uni-augsburg.de/itphy-sw-origin/forktps

[ITensor]: https://github.com/ITensor/ITensor
