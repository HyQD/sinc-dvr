# Cartesian sinc discrete-variable representation (DVR) matrix elements

![](https://github.com/HyQD/sinc-dvr/actions/workflows/python-package.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Implementation of the 3D Cartesian sinc DVR basis from the paper by [Jones et al.](https://doi.org/10.1080/00268976.2016.1176262).


## Testing different device geometries

To test locally different device geometries, the `XLA_FLAGS` from
[jax](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#aside-hosts-and-devices-in-jax)
can be used.
For example, to emulate 9 devices do
```bash
export XLA_FLAGS='--xla_force_host_platform_device_count=9'
```
or, alternatively at the top of the script with
```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=9'
```


## Installation

As Jax requires a specification of the device to be used, we have included this
in the installation of `sinc-dvr` as well.
For a local installation using jaxlib on the CPU this consists of:
```bash
pip install -e ".[cpu]"
```
Instead of `cpu` the same flags as for jax can be specified.
For example, using CUDA version 12 with binaries built from pip can be installed via:
```bash
pip install -e ".[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Here the link after `-f` is needed by Jax to find the correct CUDA components.


## Testing
To run the tests call:
```bash
python -m unittest
```
Note that the tests emulate several devices using the `XLA_FLAGS` environment
variable from above.
This does not seem to supported on the GPU, so the tests should be run using
the CPU.
