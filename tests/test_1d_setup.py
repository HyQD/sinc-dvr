import os
import math

# Note that device shape is global across testing.
# That is, the XLA_FLAGS can only be set once.
# This means that the product of the device shape across different tests
# scripts must be equal.

device_shape = (3,)
num_devices = math.prod(device_shape)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"


import unittest
import functools
import jax
import jax.numpy as jnp
import jax.sharding
from jax.experimental import mesh_utils

import sinc_dvr as sd


@jax.jit
def shielded_coulomb(x_1, x_2, strength, shielding):
    return strength / jnp.sqrt((x_1**2 - x_2**2) + shielding**2)


class Setup1DFuncTests(unittest.TestCase):
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

        solver = jax.jit(
            functools.partial(
                jax.scipy.sparse.linalg.cg,
                tol=1e-3,
            ),
            static_argnums=(0,),
        )

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
            steps,
            u_fft_circ,
            kind="d",
        )
        u_exchange = sd.get_two_body_toeplitz_matvec_operator(
            inds,
            steps,
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
