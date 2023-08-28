import os
import math

# Note that device shape is global across testing.
# That is, the XLA_FLAGS can only be set once.
# This means that the product of the device shape across different tests
# scripts must be equal.

device_shape = (1, 1, 3)
num_devices = math.prod(device_shape)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"

import unittest
import jax
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

from sinc_dvr import SincDVR
from sinc_dvr.basis import fft_matvec_solution


class Setup3DTests(unittest.TestCase):
    def test_3d_setup(self):
        sd = SincDVR(
            positive_extent=(0.9, 2.2, 5.0),
            steps=(0.1, 0.2, 0.3),
            device_shape=device_shape,
            build_t_inv=True,
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
        ident = (
            jnp.einsum("ip, pjk -> ijk", sd.t_x, sd.t_inv)
            + jnp.einsum("jp, ipk -> ijk", sd.t_y, sd.t_inv)
            + jnp.einsum("kp, ijp -> ijk", sd.t_z, sd.t_inv)
        )

        i, j, k = (
            sd.inds[0][:, None, None],
            sd.inds[1][None, :, None],
            sd.inds[2][None, None, :],
        )
        test_ident = jnp.zeros_like(ident)
        test_ident = test_ident.at[(i == j) & (i == k) & (j == k) & (i == 0)].set(1.0)

        assert jnp.allclose(ident, test_ident, atol=1e-3)
        assert jnp.allclose(
            ident, sd.get_kinetic_matvec_operator()(sd.t_inv).reshape(sd.grid_shape)
        )
        assert abs(jnp.sum(jnp.abs(ident)) - 1) < 1e-3

    def test_matvec_kinetic_mels(self):
        sd = SincDVR(
            positive_extent=(0.9, 2.2, 5.0),
            steps=(0.1, 0.2, 0.3),
            device_shape=device_shape,
            build_t_inv=False,
        )

        c = jax.random.normal(jax.random.PRNGKey(1), sd.grid_shape)

        res = (
            jnp.einsum("ip, pjk -> ijk", sd.t_x, c)
            + jnp.einsum("jp, ipk -> ijk", sd.t_y, c)
            + jnp.einsum("kp, ijp -> ijk", sd.t_z, c)
        )

        res_fft = sd.matvec_kinetic(c.ravel())
        mvk_func = sd.get_kinetic_matvec_operator()

        assert jnp.allclose(res.ravel(), res_fft)
        assert jnp.allclose(res.ravel(), mvk_func(c.ravel()))

    def test_t_inv_fft(self):
        sd = SincDVR(
            positive_extent=(0.9, 2.2, 1.5),
            steps=(0.1, 0.2, 0.3),
            device_shape=device_shape,
            build_t_inv=True,
        )

        c = jnp.array([0.0, 0.0, 0.0])
        shift = jnp.array([sd.x[0], sd.y[0], sd.z[0]])
        t_inv_from_fft = fft_matvec_solution(
            sd.t_inv_fft_circ,
            (
                chi := sd.evaluate_basis_functions(
                    c + shift,
                    [
                        sd.x[:, None, None],
                        sd.y[None, :, None],
                        sd.z[None, None, :],
                    ],
                )
            ),
        )

        # Test the shifting of the center by instead shifting the grid
        z = [sd.grid_shape[i] // 2 for i in range(3)]
        assert (
            abs(
                jnp.sum(jnp.abs(chi))
                - sd.evaluate_basis_functions(c, [sd.x[z[0]], sd.y[z[1]], sd.z[z[2]]])
            )
            < 1e-12
        )

        assert jnp.allclose(sd.t_inv / jnp.sqrt(sd.tot_weight), t_inv_from_fft)

        sd.construct_r_inv_potentials([c, jnp.array([sd.steps[0], 0.0, 0.0])], [-1, 1])

        assert jnp.allclose(
            -2 * jnp.pi * sd.t_inv.ravel() / sd.tot_weight, sd.r_inv_potentials[0]
        )
        # Check that maximum of (positive) Coulomb potential is at [dx, 0, 0]
        x_max = jnp.argmax(
            sd.r_inv_potentials[1].real.reshape(sd.grid_shape)[:, z[1], z[2]]
        )
        print(x_max, sd.x[x_max], sd.steps[0])
        assert (
            sd.x[
                jnp.argmax(
                    sd.r_inv_potentials[1].real.reshape(sd.grid_shape)[:, z[1], z[2]]
                )
            ]
            == sd.steps[0]
        )

    def test_r_inv_potentials(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(9, 11, 13),
            device_shape=device_shape,
            build_t_inv=True,
            n_in_factor=(5, 7, 9),
            n_out_factor=(27, 29, 31),
        ).construct_r_inv_potentials(
            centers=[jnp.array([0.0, 0.0, 0.0]), jnp.array([0.5, 0.3, -0.5])],
            charges=[-1.0, 2.0],
        )

        assert sd.r_inv_potentials[0].shape == (math.prod(sd.grid_shape),)
        assert sd.r_inv_potentials[1].shape == (math.prod(sd.grid_shape),)

    def test_coulomb_interaction_operators(self):
        sd = SincDVR(
            num_dim=3,
            steps=(0.1, 0.2, 0.3),
            element_factor=(9, 11, 13),
            device_shape=device_shape,
            build_t_inv=True,
            n_in_factor=(5, 7, 9),
            n_out_factor=(27, 29, 31),
        )
        coulomb_d = sd.get_coulomb_interaction_matvec_operator(-1, -1, "d")
        coulomb_e = sd.get_coulomb_interaction_matvec_operator(-1, -1, "e")

        c = jax.random.normal(jax.random.PRNGKey(2), (math.prod(sd.grid_shape), 3))

        c_d = coulomb_d(c[:, 0].conj(), c[:, 1], c[:, 2])
        c_e = coulomb_e(c[:, 0].conj(), c[:, 2], c[:, 1])

        assert jnp.allclose(c_d, c_e, atol=1e-3)
