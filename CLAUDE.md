# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

University course repository (MYCHP203 - Parallel Optimization Techniques) containing lab exercises and a main project, all focused on HPC/parallel computing in C/C++.

## Build Commands

All C/C++ code uses CMake (minimum 3.25). Each subdirectory (`lab1/*`, `lab2/*`, `project/`) is an independent CMake project — configure and build from within that directory.

```bash
# Generic pattern for any exercise or the project
cmake -B build
cmake --build build

# For release/optimized builds
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Lab2 matrix-product requires Kokkos flags
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON
```

### Project (LBM solver)

```bash
# Build simulation binary
cd project
cmake -B build
cmake --build build -t top.lbm-exe

# Build display helper
cmake --build build -t top.display

# Run simulation (MPI required)
mpirun -np <N> ./build/top.lbm-exe config.txt

# Validate results against reference
uv pip install -e .
lbm-viz --check ref_results.raw results.raw

# Generate visualization
lbm-viz --generate-gif results.raw output.gif
```

## Pre-commit Hooks

Active hooks: trailing whitespace trimming, EOF fixer, clang-format (v22.1.1) for C/C++, gersemi for CMake formatting. Run `pre-commit install` to set up.

## Code Style

- C/C++: LLVM-based clang-format (120 col limit, 2-space indent, right-qualified, block-indent after open bracket). See `.clang-format`.
- CMake: gersemi formatter (120 col, 2-space indent, favour-inlining for lists).

## Architecture

### Project — Hybrid MPI+OpenMP D2Q9 Lattice Boltzmann Solver

Located in `project/`. Simulates Karman vortex street flow. Key structure:

- **`include/lbm/`**: All public headers
  - `config.hpp` — global config struct (`lbm_gbl_config`) loaded from text file, plus macros for mesh dimensions/physics params
  - `structures.hpp` — core data types: `Mesh` (cell array), `lbm_mesh_type_t` (cell classification), file I/O structs
  - `communications.hpp` — MPI domain decomposition (`lbm_comm_t`), halo exchange, frame gathering
  - `physics.hpp` — collision, propagation, boundary conditions (Zou/He, bounce-back)
  - `initialization.hpp` — initial state setup
- **`src/lbm/`**: Implementation of the library (`top.lbm-lib`, shared library)
- **`src/bin/main.cpp`**: MPI main loop — init, time-stepping (special_cells -> collision -> halo_exchange -> propagation), output
- **`src/bin/display.cpp`**: Standalone binary for rendering `.raw` output files
- **`src/lbm_viz/`**: Python visualization tool (`lbm-viz`), installed via `uv pip install -e .`
- **`cmake/third-party.cmake`**: Generates `tpl.hpp` and `tpl_loader.hpp` at configure time

The simulation uses a D2Q9 lattice (2 dimensions, 9 velocity directions). Domain is decomposed across MPI ranks; each rank owns a horizontal strip with ghost/phantom cells for neighbor communication.

### Lab Exercises

- **`lab1/bugs/`**: Standalone C file with intentional bugs to find
- **`lab1/mol-dyn/`**: Molecular dynamics simulation (C/C++ with {fmt}), PRNG library in C
- **`lab1/stream/`**: STREAM memory bandwidth benchmark (C++ with {fmt})
- **`lab1/saxpy/`**: SAXPY kernel (standalone C file, no CMake)
- **`lab1/vector/`**: Vector operations (header-only, no CMake)
- **`lab2/matrix-product/`**: Matrix multiplication with Kokkos + OpenMP
- **`lab2/mesh/`**: Mesh computation exercise with OpenMP
- **`lab2/branch-predict/`**: Branch prediction example (standalone C++ file)

## Dependencies

- MPI 3.0+ and OpenMP 4.0+ (project)
- {fmt} 11.1.4 (fetched via CMake FetchContent in labs)
- Kokkos 4.6.00 (fetched via CMake FetchContent in lab2/matrix-product)
- Python 3.10+, uv, gnuplot (project visualization)
