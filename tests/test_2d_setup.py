import os
import math

device_shape = (2, 3)
num_devices = math.prod(device_shape)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"

import unittest
import jax
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

from sinc_dvr import SincDVR


class Setup2DTests(unittest.TestCase):
    def test_2d_setup(self):
        sd = SincDVR(
            num_dim=2,
            steps=(0.1, 0.2),
            element_factor=(10, 11),
            device_shape=device_shape,
        )

        # Check that grid is equal on both sides of zero
        for axis_name in ["x", "y"]:
            grid = getattr(sd, axis_name)
            l = len(grid)

            if l % 2 == 0:  # Even number of grid points, no zero
                assert jnp.sum(jnp.abs(grid[: l // 2][::-1] + grid[l // 2 :])) < 1e-12
            else:  # Odd number of grid points, zero included
                assert (
                    jnp.sum(jnp.abs(grid[: l // 2][::-1] + grid[l // 2 + 1 :])) < 1e-12
                )
                assert jnp.abs(grid[l // 2]) < 1e-12

    def test_matvec_kinetic_mels(self):
        sd = SincDVR(
            num_dim=2,
            steps=(0.1, 0.2),
            element_factor=(10, 11),
            device_shape=device_shape,
        )

        c = jax.random.normal(jax.random.PRNGKey(1), sd.element_shape)

        res = jnp.einsum("ip, pj -> ij", sd.t_x, c) + jnp.einsum(
            "jp, ip -> ij", sd.t_y, c
        )

        res_fft = sd.matvec_kinetic(c.ravel())
        mvk_func = sd.get_kinetic_matvec_operator()

        assert jnp.allclose(res.ravel(), res_fft)
        assert jnp.allclose(res.ravel(), mvk_func(c.ravel()))
