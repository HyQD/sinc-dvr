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
import functools
import jax
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

import sinc_dvr.func_basis as sdf


def get_p_matvec_operator_2(inds, steps, axis):
    shape = [len(ind.ravel()) for ind in inds]
    dim = len(inds)
    assert dim <= 3
    assert axis < len(inds)

    p = sdf.setup_p_1d(
        inds[axis].ravel()[:, None], inds[axis].ravel()[None, :], steps[axis]
    )

    p_einsum = ["ip", "jp", "kp"][axis]
    c_einsum = [
        ["p"],
        ["pj", "ip"],
        ["pjk", "ipk", "ijp"],
    ][
        dim - 1
    ][axis]
    o_einsum = ["i", "ij", "ijk"][dim - 1]

    @jax.jit
    def matvec_p(
        c, p=p, shape=shape, p_einsum=p_einsum, c_einsum=c_einsum, o_einsum=o_einsum
    ):
        return jnp.einsum(
            f"{p_einsum}, {c_einsum} -> {o_einsum}", p, c.reshape(shape)
        ).ravel()

    return matvec_p


class Setup3DFuncTests(unittest.TestCase):
    def test_3d_setup(self):
        steps = 0.1, 0.2, 0.3
        positive_extent = 1.0, 1.1, 1.2
        inds = sdf.get_oinds(positive_extent, steps, verbose=True)
        shape = [len(ind.ravel()) for ind in inds]

        solver = jax.jit(
            functools.partial(
                jax.scipy.sparse.linalg.cg,
                tol=1e-3,
            ),
            static_argnums=(0,),
        )

        print("Setting up t_fft_circ")
        t_fft_circ = sdf.get_t_fft_circ(inds, steps)
        t_op = sdf.get_kinetic_matvec_operator(t_fft_circ)
        print("Setting up t_inv_fft_circ")
        t_inv_fft_circ = sdf.get_t_inv_fft_circ(inds, steps, t_op, solver=solver)
        v_pot_func = sdf.get_r_inv_potential_function(inds, steps, t_inv_fft_circ)
        print("Setting up v_op_1")
        v_op_1 = sdf.get_position_dependent_matvec_operator(
            inds,
            v_pot_func(jnp.array([0.0, 0.0, -0.5]), -1),
        )
        print("Setting up v_op_2")
        v_op_2 = sdf.get_position_dependent_matvec_operator(
            inds,
            v_pot_func(jnp.array([0.0, 0.0, 0.5]), -1),
        )
        print("Setting up Coulomb direct")
        u_d_op = sdf.get_coulomb_interaction_matvec_operator(
            inds,
            steps,
            t_inv_fft_circ,
            -1.0,
            -1.0,
            "d",
        )
        print("Setting up Coulomb exchange")
        u_e_op = sdf.get_coulomb_interaction_matvec_operator(
            inds,
            steps,
            t_inv_fft_circ,
            -1.0,
            -1.0,
            "e",
        )
        p_x, p_y, p_z = [
            sdf.get_p_matvec_operator(inds, steps, i) for i in range(len(steps))
        ]

        x, y, z = [ind * dw for ind, dw in zip(inds, steps)]
        ho_pot_op = sdf.get_position_dependent_matvec_operator(
            inds,
            jax.jit(lambda x=x, y=y, omega=2: 0.5 * omega**2 * (x**2 + y**2)),
        )
        x_op = sdf.get_position_dependent_matvec_operator(inds, x)
        y_op = sdf.get_position_dependent_matvec_operator(inds, y)

        t = 1
        c = jax.random.normal(jax.random.PRNGKey(1), (math.prod(shape),), dtype=complex)

        # Compare the two ways of computing the momentum matrix vector product
        p2_x, p2_y, p2_z = [
            get_p_matvec_operator_2(inds, steps, i) for i in range(len(steps))
        ]
        assert jnp.allclose(p_x(c), p2_x(c))
        assert jnp.allclose(p_y(c), p2_y(c))
        assert jnp.allclose(p_z(c), p2_z(c))

        laser = jax.jit(
            lambda t, x=x, shape=shape, omega=1, k=2, E0=1: E0
            * jnp.sin(omega * t - k * x)
        )

        laser_op = sdf.get_position_dependent_matvec_operator(inds, laser)

        @jax.jit
        def H(
            t,
            c,
            t_op=t_op,
            v_op_1=v_op_1,
            v_op_2=v_op_2,
            u_d_op=u_d_op,
            u_e_op=u_e_op,
            ho_pot_op=ho_pot_op,
            x_op=x_op,
            y_op=y_op,
            p_x=p_x,
            p_y=p_y,
            p_z=p_z,
            laser_op=laser_op,
        ):
            return (
                t_op(c)  # Kinetic energy operator
                + (v_op_1(c) + v_op_2(c))  # 1/r-potentials
                + u_d_op(c, c.conj(), c)  # Coulomb direct
                + u_e_op(c, c.conj(), c)  # Coulomb exchange
                + ho_pot_op(c)  # Static magnetic field
                + (x_op(p_y(c)) - y_op(p_x(c)))  # l_z operator
                + laser_op(p_z(c), t)
            )

        res = H(t, c)
        assert len(res) == len(c.ravel())
