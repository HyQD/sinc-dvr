import os
import math

# Note that device shape is global across testing.
# That is, the XLA_FLAGS can only be set once.
# This means that the product of the device shape across different tests
# scripts must be equal.

device_shape = (3,)
num_devices = math.prod(device_shape)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"


# import jax.config
# jax.config.update("jax_enable_x64", True)


import unittest
import functools
import jax
import jax.numpy as jnp
import jax.sharding
from jax.experimental import mesh_utils


import numpy as np

import sinc_dvr as sd


@jax.jit
def shielded_coulomb(x_1, x_2, strength, shielding):
    return strength / jnp.sqrt((x_1 - x_2) ** 2 + shielding**2)


class Setup1DFuncTests(unittest.TestCase):
    def test_toeplitz_structure_of_shielded_coulomb_interaction(self):
        steps = (0.1,)
        positive_extent = (5.0,)

        inds = sd.get_oinds(positive_extent, steps, verbose=True)
        shape = [len(ind.ravel()) for ind in inds]

        (x,) = [ind * dw for ind, dw in zip(inds, steps)]

        u = shielded_coulomb(x[None, :], x[:, None], 1, 0.25)
        u_tilde = shielded_coulomb(x, x.ravel()[0], 1, 0.25)  # First column of u
        u_tilde_2 = shielded_coulomb(x.ravel()[0], x, 1, 0.25)  # First row of u
        # Check that u_tilde equals the first column of u
        np.testing.assert_allclose(u[:, 0], u_tilde)
        # Check that u is Hermitian (symmetric in our case)
        np.testing.assert_allclose(u, u.conj().T)
        # If u is Hermitian, then the first row should equal the first column
        # complex conjugated
        np.testing.assert_allclose(u_tilde, u_tilde_2.conj())

        # Make the first row work with Python indexing. That is, we wish
        # u_tilde_2[0] == u_tilde[0], and u_tilde_2[-1] to be the first column
        # in u_tilde_2. This is to fit with the definition of a Toeplitz matrix
        # as given here: https://en.wikipedia.org/wiki/Toeplitz_matrix
        u_tilde_2 = jnp.concatenate([jnp.array([u_tilde_2[0]]), u_tilde_2[1:][::-1]])

        for i in range(u.shape[0]):
            for j in range(i, u.shape[1]):

                if i + 1 < u.shape[0] and j + 1 < u.shape[1]:
                    # Check that u has constant diagonals
                    np.testing.assert_allclose(u[i, j], u[i + 1, j + 1], atol=1e-5)

                # Check that the Toeplitz definition A_{i, j} = a_{i - j} holds
                # with 'A' being the full 'u' and 'a' being 'u_tilde'.
                # NOTE: This equality is not directly comparable due to how
                # Python handles indexing. Also, a general Toeplitz matrix is
                # not symmetric such that the negative indices should
                # correspond to the first row of u. We perform several versions
                # of this test below.

                # As u is symmetric we have that the first row and the first
                # column are the same. As such, 'a_{i} = a_{-i}' with 'a' being
                # 'u_tilde'. We therefore have 'a_{i} = a_{-i} = a_{|i|}'.
                np.testing.assert_allclose(u[i, j], u_tilde[abs(i - j)], atol=1e-5)

                # The more "natural" Toeplitz test. If i - j >= 0, compare u to
                # the first column of u, u_tilde. Otherwise, compare u to the
                # first row of u, u_tilde_2.conj().
                np.testing.assert_allclose(
                    u[i, j],
                    u_tilde[i - j] if i - j >= 0 else u_tilde_2[i - j].conj(),
                    atol=1e-5,
                )

    def test_1d_shielded_coulomb_interaction_single_orbital(self):
        # In this test we compare if the "full" shielded Coulomb interaction
        # elements (i.e., the rank-2 tensor form of the interaction) gives the
        # same result as when using the FFT-solution in circulant form.

        steps = (0.1,)
        positive_extent = (5.0,)

        inds = sd.get_oinds(positive_extent, steps, verbose=True)
        shape = [len(ind.ravel()) for ind in inds]

        (x,) = [ind * dw for ind, dw in zip(inds, steps)]

        u = shielded_coulomb(x[None, :], x[:, None], 1, 0.25)
        u_tilde = shielded_coulomb(x.ravel(), x.ravel()[0], 1, 0.25)
        np.testing.assert_allclose(u[:, 0], u_tilde, atol=1e-5)
        np.testing.assert_allclose(u, u.conj().T)

        u_fft_circ = sd.get_fft_embedded_circulant(u_tilde)

        u_direct = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="d",
        )
        u_exchange = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="e",
        )

        c = jax.random.normal(
            jax.random.PRNGKey(6), (math.prod(shape),), dtype=complex
        ) + 1j * jax.random.normal(
            jax.random.PRNGKey(5), (math.prod(shape),), dtype=complex
        )

        # Note the ordering!
        res = (u @ (c.conj() * c)) * c
        res_2 = u_direct(c, c.conj(), c)
        res_3 = u_exchange(c, c.conj(), c)

        # NOTE: These tolerances might seem low, but using f64 instead of f32
        # we can remove the tolerance completely. In other words, this seems to
        # be a passing test.
        np.testing.assert_allclose(res, res_2, atol=1e-3)
        # Exchange and direct are the same when all three coefficients are the
        # same.
        np.testing.assert_allclose(res, res_3, atol=1e-3)
        np.testing.assert_allclose(c.conj() @ res, c.conj() @ res_2, atol=1e-3)

    def test_1d_shielded_coulomb_interaction_multiple_orbitals(self):
        # In this test we compare if the "full" shielded Coulomb interaction
        # elements (i.e., the rank-2 tensor form of the interaction) gives the
        # same result as when using the FFT-solution in circulant form.

        num_orbitals = 4
        num_occupied = 2
        steps = (0.1,)
        positive_extent = (5.0,)

        inds = sd.get_oinds(positive_extent, steps, verbose=True)
        shape = [len(ind.ravel()) for ind in inds]

        (x,) = [ind * dw for ind, dw in zip(inds, steps)]

        u = shielded_coulomb(x[None, :], x[:, None], 1, 0.25)
        u_tilde = shielded_coulomb(x.ravel(), x.ravel()[0], 1, 0.25)
        np.testing.assert_allclose(u[:, 0], u_tilde, atol=1e-5)
        np.testing.assert_allclose(u, u.conj().T)

        u_fft_circ = sd.get_fft_embedded_circulant(u_tilde)

        u_direct = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="d",
        )
        u_exchange = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="e",
        )

        # Build vectorized versions of the direct and exchange integrals
        u_direct_v = jax.vmap(
            jax.vmap(
                u_direct,
                (None, 1, 1),
                1,
            ),
            (1, None, None),
            1,
        )
        u_exchange_v = jax.vmap(
            jax.vmap(
                u_exchange,
                (None, 1, 1),
                1,
            ),
            (1, None, None),
            1,
        )

        cs = jax.random.normal(
            jax.random.PRNGKey(3), (math.prod(shape), num_orbitals), dtype=complex
        ) + 1j * jax.random.normal(
            jax.random.PRNGKey(2), (math.prod(shape), num_orbitals), dtype=complex
        )

        res_d, res_e, res_d_2, res_e_2 = [], [], [], []

        # Test using all orbitals
        for i in range(cs.shape[1]):
            c = cs[:, i]
            _res_d, _res_e, _res_d_2, _res_e_2 = [], [], [], []

            for j in range(min(num_occupied, cs.shape[1])):
                d = cs[:, j]

                # Note the ordering!
                _res_d.append((u @ (d.conj() * d)) * c)
                _res_e.append((u @ (d.conj() * c)) * d)
                _res_d_2.append(u_direct(c, d.conj(), d))
                _res_e_2.append(u_exchange(c, d.conj(), d))

                # Check that the exchange and the direct term gives the same
                # when we use the same orbital in all three positions.
                if i == j:
                    np.testing.assert_allclose(_res_d[-1], _res_e[-1])
                    np.testing.assert_allclose(_res_d[-1], _res_d_2[-1], atol=1e-3)
                    np.testing.assert_allclose(_res_d[-1], _res_e_2[-1], atol=1e-3)

            np.testing.assert_allclose(sum(_res_d), sum(_res_d_2), atol=1e-3)
            np.testing.assert_allclose(sum(_res_e), sum(_res_e_2), atol=1e-3)

            res_d.append(sum(_res_d))
            res_e.append(sum(_res_e))
            res_d_2.append(sum(_res_d_2))
            res_e_2.append(sum(_res_e_2))

        res_d = jnp.array(res_d).T
        res_e = jnp.array(res_e).T
        res_d_2 = jnp.array(res_d_2).T
        res_e_2 = jnp.array(res_e_2).T

        assert res_d.shape == cs.shape
        assert res_e.shape == cs.shape
        assert res_d_2.shape == cs.shape
        assert res_e_2.shape == cs.shape

        res_d_v = jnp.sum(
            u_direct_v(
                cs, cs[:, slice(0, num_occupied)].conj(), cs[:, slice(0, num_occupied)]
            ),
            axis=2,
        )
        res_e_v = jnp.sum(
            u_exchange_v(
                cs, cs[:, slice(0, num_occupied)].conj(), cs[:, slice(0, num_occupied)]
            ),
            axis=2,
        )
        assert res_d_v.shape == cs.shape
        assert res_e_v.shape == cs.shape

        np.testing.assert_allclose(res_d_v, res_d, atol=1e-3)
        np.testing.assert_allclose(res_d_v, res_d_2, atol=1e-3)
        np.testing.assert_allclose(res_e_v, res_e, atol=1e-3)
        np.testing.assert_allclose(res_e_v, res_e_2, atol=1e-3)

    def test_1d_setup_sharded(self):
        axis_names = ["x"]
        mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh(device_shape), axis_names=axis_names
        )
        spec = jax.sharding.PartitionSpec(*axis_names)

        steps = (0.1,)
        positive_extent = (1.0,)

        inds = sd.get_oinds_sharded(positive_extent, steps, device_shape, verbose=True)
        inds = [
            jax.device_put(
                ind.ravel(),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name)),
            ).reshape(ind.shape)
            for ind, axis_name in zip(inds, axis_names)
        ]

        assert all(len(ind.devices()) == num_devices for ind in inds)

        shape = [len(ind.ravel()) for ind in inds]

        print("Setting up t_fft_circ")
        t_fft_circ = sd.get_t_fft_circ(inds, steps)
        t_op = sd.get_kinetic_matvec_operator(t_fft_circ)

        (x,) = [ind * dw for ind, dw in zip(inds, steps)]
        ho_pot_op = sd.get_position_dependent_matvec_operator(
            inds,
            jax.jit(lambda x=x, omega=2: 0.5 * omega**2 * x**2),
        )

        u_tilde = shielded_coulomb(x, x.ravel()[0], 1, 0.25)
        u_fft_circ = sd.get_fft_embedded_circulant(u_tilde)

        u_direct = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="d",
        )
        u_exchange = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            u_fft_circ,
            kind="e",
        )

        t = 1
        c = jax.random.normal(jax.random.PRNGKey(1), (math.prod(shape),), dtype=complex)

        laser = jax.jit(
            lambda t, x=x, shape=shape, omega=1, k=2, E0=1: E0
            * jnp.sin(omega * t - k * x)
        )

        laser_op = sd.get_position_dependent_matvec_operator(inds, laser)

        @jax.jit
        def H(
            t,
            c,
            t_op=t_op,
            ho_pot_op=ho_pot_op,
            u_direct=u_direct,
            u_exchange=u_exchange,
            laser_op=laser_op,
        ):
            return (
                t_op(c)  # Kinetic energy operator
                + ho_pot_op(c)  # Static magnetic field
                + 2 * u_direct(c, c.conj(), c)  # Direct shielded Coulomb interaction
                - u_exchange(c, c.conj(), c)  # Exchange shielded Coulomb interaction
                + laser_op(c, t)  # Dipole laser field
            )

        res = H(t, c)
        assert len(res) == len(c.ravel())
