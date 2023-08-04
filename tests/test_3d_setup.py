import os
import math

# Note that device shape is global across testing.
# That is, the XLA_FLAGS can only be set once.
# This means that the product of the device shape across different tests
# scripts must be equal.

device_shape = (1, 2, 3)
num_devices = math.prod(device_shape)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"

import unittest
import jax
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

from sinc_dvr import SincDVR


class Setup3DTests(unittest.TestCase):
    def test_3d_setup(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=device_shape,
            build_t_inv=True,
            n_in_factor=(6, 7, 9),
            n_out_factor=(28, 29, 31),
        )

        # Check that grid is equal on both sides of zero
        for axis_name in ["x", "y", "z"]:
            grid = getattr(sd, axis_name)
            l = len(grid)

            if l % 2 == 0:  # Even number of grid points, no zero
                assert jnp.sum(jnp.abs(grid[: l // 2][::-1] + grid[l // 2 :])) < 1e-12
            else:  # Odd number of grid points, zero included
                assert (
                    jnp.sum(jnp.abs(grid[: l // 2][::-1] + grid[l // 2 + 1 :])) < 1e-12
                )
                assert jnp.abs(grid[l // 2]) < 1e-12

        # Check that t @ t_inv gives the identity
        # ident = (
        #     jnp.einsum("ip, pjk -> ijk", sd.t_x, sd.t_inv)
        #     + jnp.einsum("jp, ipk -> ijk", sd.t_y, sd.t_inv)
        #     + jnp.einsum("kp, ijp -> ijk", sd.t_z, sd.t_inv)
        # )
        # Note: This is only approximately true when using only an odd number
        # of elements

    def test_matvec_kinetic_mels(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=device_shape,
            build_t_inv=False,
        )

        c = jax.random.normal(jax.random.PRNGKey(1), sd.element_shape)

        res = (
            jnp.einsum("ip, pjk -> ijk", sd.t_x, c)
            + jnp.einsum("jp, ipk -> ijk", sd.t_y, c)
            + jnp.einsum("kp, ijp -> ijk", sd.t_z, c)
        )

        res_fft = sd.matvec_kinetic(c.ravel())
        mvk_func = sd.get_kinetic_matvec_operator()

        assert jnp.allclose(res.ravel(), res_fft)
        assert jnp.allclose(res.ravel(), mvk_func(c.ravel()))

    def test_r_inv_potentials(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=device_shape,
            build_t_inv=True,
            n_in_factor=(6, 7, 9),
            n_out_factor=(28, 29, 31),
        ).construct_r_inv_potentials(
            centers=[jnp.array([0.0, 0.0, 0.0]), jnp.array([0.5, 0.3, -0.5])],
            charges=[-1.0, 2.0],
        )

        assert sd.r_inv_potentials[0].shape == tuple(sd.element_shape)
        assert sd.r_inv_potentials[1].shape == tuple(sd.element_shape)

    def test_coulomb_interaction_operators(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=device_shape,
            build_t_inv=True,
            n_in_factor=(6, 7, 9),
            n_out_factor=(28, 29, 31),
        )
        coulomb_d = sd.get_coulomb_interaction_matvec_operator(-1, -1, "d")
        coulomb_e = sd.get_coulomb_interaction_matvec_operator(-1, -1, "e")

        c = jax.random.normal(jax.random.PRNGKey(2), (math.prod(sd.element_shape), 3))

        c_d = coulomb_d(c[:, 0].conj(), c[:, 1], c[:, 2])
        c_e = coulomb_e(c[:, 0].conj(), c[:, 2], c[:, 1])

        assert jnp.allclose(c_d, c_e, atol=1e-3)
