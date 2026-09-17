# JAX HPC Apptainer

A reproducible GPU-enabled JAX environment for NVIDIA-based HPC systems using Docker and Apptainer.

## Overview

This repository provides a reproducible container environment for running JAX-based scientific computing workloads on HPC systems with NVIDIA GPUs.

The environment was built for `linux/amd64` and has been successfully tested with:

- Apptainer
- NVIDIA L20 GPU
- JAX GPU backend
- 64-bit (`float64`) JAX computations

## Verified software environment

- Python 3.11.16
- JAX 0.10.2
- jaxlib 0.10.2
- NumPy 2.4.6
- Numba 0.67.0
- Matplotlib 3.11.2
- PyVista 0.49.0
- CUDA 12 JAX runtime

The complete Python environment is pinned in:

`requirements-verified.txt`

## Platform requirements

The prebuilt container is intended for systems with:

- Linux x86_64
- NVIDIA GPU
- Compatible NVIDIA driver
- Apptainer or Singularity

Slurm is not required by the container itself, but example Slurm scripts are provided for HPC usage.


## Quick start

### 1. Test the container on an NVIDIA GPU

On a GPU compute node, run:

```bash
apptainer exec --nv \
    jax-hpc-apptainer_1.0.0.sif \
    python examples/jax_gpu_test.py
```

A successful run should report a GPU backend and at least one CUDA device, for example:

```text
Default backend: gpu
Devices: [CudaDevice(id=0)]

Calculation completed.
dtype: float64
shape: (2000, 2000)
device: cuda:0
```

The test performs an actual double-precision (`float64`) matrix multiplication rather than only checking whether JAX can detect the GPU.

### 2. Run with Slurm

A generic Slurm example is provided in:

```text
slurm/jax_gpu_test.slurm
```

Before submitting the job, edit the following paths in the Slurm script:

```bash
IMAGE=/path/to/jax-hpc-apptainer_1.0.0.sif
PROJECT_DIR=/path/to/jax-hpc-apptainer
```

Depending on your HPC system, you may also need to add cluster-specific Slurm options such as:

```bash
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
```

Then submit the job with:

```bash
sbatch slurm/jax_gpu_test.slurm
```

### 3. GPU access inside Apptainer

The container must be launched with NVIDIA GPU support enabled:

```bash
apptainer exec --nv ...
```

The `--nv` option exposes the NVIDIA GPU devices and compatible host NVIDIA driver libraries to the container.

The prebuilt image targets:

- Linux `x86_64`
- NVIDIA GPUs
- Apptainer or Singularity
- Compatible NVIDIA drivers

It is not intended for ARM-based HPC nodes or AMD GPUs.


## Status

Version 1.0.0 is the first reproducible environment release.
