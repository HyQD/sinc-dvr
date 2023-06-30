# Cartesian sinc discrete-variable representation (DVR) matrix elements


## Testing different geometries

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
pip install -e ".[cuda12_pip]"
```

# We need to handle the links used by Jax (e.g., `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`).

