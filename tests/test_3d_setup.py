import unittest
import jax
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

from sinc_dvr import SincDVR


class Setup3DTests(unittest.TestCase):
    def test_3d_1device_setup(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=(1, 1, 1),
            build_t_inv=True,
            n_s=(6, 7, 9),
            n_b=(28, 29, 31),
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
            device_shape=(1, 1, 1),
            build_t_inv=False,
        )

        c = jax.random.normal(jax.random.PRNGKey(1), sd.element_shape)

        res = (
            jnp.einsum("ip, pjk -> ijk", sd.t_x, c)
            + jnp.einsum("jp, ipk -> ijk", sd.t_y, c)
            + jnp.einsum("kp, ijp -> ijk", sd.t_z, c)
        )

        res_fft = sd.matvec_kinetic(c.ravel())

        assert jnp.allclose(res.ravel(), res_fft)

    def test_r_inv_potentials(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(10, 11, 13),
            device_shape=(1, 1, 1),
            build_t_inv=True,
            # n_s=(6, 7, 9),
            # n_b=(28, 29, 31),
        )

        c = jax.random.normal(jax.random.PRNGKey(2), sd.element_shape)

        sd.construct_r_inv_potentials(
            centers=[jnp.array([0.0, 0.0, 0.0]), jnp.array([0.5, 0.3, -0.5])],
            charges=[-1.0, 2.0],
        )
