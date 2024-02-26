import math
import functools
import itertools

import jax
import jax.numpy as jnp


def get_oinds(positive_extent, steps, verbose=False):
    num_dim = len(positive_extent)
    num_points_shape = [
        # Ensure an odd number of points to include the origin
        math.ceil(pos_ext / dw) * 2 + 1
        for pos_ext, dw in zip(positive_extent, steps)
    ]
    num_points_pos_shape = [(nps - 1) / 2 for nps in num_points_shape]
    new_positive_extent = [npps * dw for npps, dw in zip(num_points_pos_shape, steps)]
    num_elements = math.prod(num_points_shape)

    if (
        any(
            [
                (new_pos_ext - pos_ext) != 0
                for new_pos_ext, pos_ext in zip(new_positive_extent, positive_extent)
            ]
        )
        and verbose
    ):
        print(
            f"Positive extent increased from: {positive_extent} to: {new_positive_extent}"
        )

    if verbose:
        print(f"Number of elements: {num_elements} distributed as: {num_points_shape}")
        # "Standard" size of a complex number
        c_size = 2 * jnp.zeros(1).dtype.itemsize
        print(
            f"Approximate size requirement for largest tensor (in GB): {2 ** num_dim * num_elements * c_size / 2 ** 30}"
        )

    slices = [slice(-npps, npps + 1) for npps in num_points_pos_shape]

    return jnp.ogrid[*slices]


def get_oinds_sharded(positive_extent, steps, device_shape, verbose=False):
    # Returns a grid that is adjusted such that it is congruent with the device shape.
    # Note that the caller needs to do the sharding.
    num_dim = len(positive_extent)
    num_points_per_device_shape = [
        # Ensure an odd number of points for an odd number of devices
        (_ := math.ceil((math.ceil(pos_ext / dw) * 2 + 1) / n_dev)) + (_ + n_dev) % 2
        for pos_ext, dw, n_dev in zip(positive_extent, steps, device_shape)
    ]
    num_points_shape = [
        nppds * n_dev for nppds, n_dev in zip(num_points_per_device_shape, device_shape)
    ]
    num_points_pos_shape = [(nps - 1) / 2 for nps in num_points_shape]
    new_positive_extent = [npps * dw for npps, dw in zip(num_points_pos_shape, steps)]
    num_elements = math.prod(num_points_shape)

    if (
        any(
            [
                (new_pos_ext - pos_ext) != 0
                for new_pos_ext, pos_ext in zip(new_positive_extent, positive_extent)
            ]
        )
        and verbose
    ):
        print(
            f"Positive extent increased from: {positive_extent} to: {new_positive_extent}"
        )

    if verbose:
        print(f"Number of elements: {num_elements} distributed as: {num_points_shape}")
        # "Standard" size of a complex number
        c_size = 2 * jnp.zeros(1).dtype.itemsize
        print(
            f"Approximate size requirement for largest tensor (in GB): {2 ** num_dim * num_elements * c_size / 2 ** 30}"
        )

    slices = [slice(-npps, npps + 1) for npps in num_points_pos_shape]

    return jnp.ogrid[*slices]


@jax.jit
def evaluate_spfs(inds, steps, position):
    return math.prod(
        [
            1 / jnp.sqrt(dw) * jnp.sinc((pos - ind * dw) / dw)
            for ind, dw, pos in zip(inds, steps, position)
        ]
    )


def get_inner_product_function(inds, steps):
    shape = [len(ind.ravel()) for ind in inds]
    tot_weight = math.prod(steps)

    @jax.jit
    def inner_product(c_conj, c_in, tot_weight=tot_weight):
        return c_conj @ c_in * tot_weight

    return inner_product


def get_position_dependent_matvec_operator(inds, pos_func):
    # Here pos_func is either a callable operator that can contain any
    # position-dependent operators, e.g., pos_func = lambda t, x=x: jnp.sin(t -
    # x), or any sum of any power of the position operators themselves, e.g.,
    # pos_func = x**2.
    shape = [len(ind.ravel()) for ind in inds]

    if callable(pos_func):

        @jax.jit
        def matvec_position_func(
            c, *func_args, shape=shape, pos_func=pos_func, **func_kwargs
        ):
            return (pos_func(*func_args, **func_kwargs) * c.reshape(shape)).ravel()

        return matvec_position_func

    @jax.jit
    def matvec_position(c, shape=shape, pos_func=pos_func):
        return (pos_func * c.reshape(shape)).ravel()

    return matvec_position


def get_kinetic_matvec_operator(t_fft_circ):
    shape = [t // 2 for t in t_fft_circ.shape]

    @jax.jit
    def matvec_kinetic(c, t_fft_circ=t_fft_circ, shape=shape):
        return fft_matvec_solution(t_fft_circ, c.reshape(shape)).ravel()

    return matvec_kinetic


def get_p_matvec_operator(inds, steps, axis):
    shape = [len(ind.ravel()) for ind in inds]
    in_shape = inds[axis].shape
    out_shape = tuple(
        in_shape[i] * (2 if i == axis else 1) for i in range(len(in_shape))
    )
    p_vec = setup_p_1d(inds[axis].ravel(), inds[axis].ravel()[0], steps[axis])
    p_fft_vec = get_fft_embedded_circulant(p_vec).reshape(out_shape)

    @jax.jit
    def matvec_p(
        c,
        p_fft_vec=p_fft_vec,
        shape=shape,
    ):
        return fft_matvec_solution(p_fft_vec, c.reshape(shape)).ravel()

    return matvec_p


def get_r_inv_potential_function(inds, steps, t_inv_fft_circ):
    # We have to subtract the edge from the centers
    # This is due to the zero location, i.e., the grid point that equals
    # the value zero, is located in the middle of the grid and hence has an
    # index that is not zero.
    shift = jnp.array([ind.ravel()[0] * dw for ind, dw in zip(inds, steps)])

    @jax.jit
    def r_inv_potential(
        center,
        charge,
        inds=inds,
        steps=steps,
        t_inv_fft_circ=t_inv_fft_circ,
        shift=shift,
    ):
        """
        Parameters
        ----------
        center : jax.Array
            The coordinates of the center of the potential.
        charge : float
            The total charge experienced by a particle interacting with the
            potential. Negative for an attractive potential, and positive for a
            repulsive potential.
        """
        return (
            2
            * jnp.pi
            / jnp.sqrt(math.prod(steps))
            * charge
            * fft_matvec_solution(
                t_inv_fft_circ, evaluate_spfs(inds, steps, center + shift)
            )
        )

    return r_inv_potential


def get_two_body_toeplitz_matvec_operator(inds, op_fft_circ, kind):
    assert kind in ["d", "e"], f"kind must either be 'd' for direct or 'e' for exchange"

    shape = [len(ind.ravel()) for ind in inds]

    @jax.jit
    def matvec_direct(
        c,
        d_conj,
        d,
        op_fft_circ=op_fft_circ,
        shape=shape,
    ):
        return fft_matvec_solution(op_fft_circ, (d_conj * d).reshape(shape)).ravel() * c

    @jax.jit
    def matvec_exchange(
        c,
        d_conj,
        d,
        op_fft_circ=op_fft_circ,
        shape=shape,
    ):
        return fft_matvec_solution(op_fft_circ, (d_conj * c).reshape(shape)).ravel() * d

    return matvec_direct if kind == "d" else matvec_exchange


def get_coulomb_interaction_matvec_operator(
    inds, steps, t_inv_fft_circ, charge_1, charge_2, kind
):
    assert kind in ["d", "e"], f"kind must either be 'd' for direct or 'e' for exchange"

    if kind == "e":
        assert charge_1 == charge_2

    shape = [len(ind.ravel()) for ind in inds]

    @jax.jit
    def matvec_direct(
        c,
        d_conj,
        d,
        charge_1=charge_1,
        charge_2=charge_2,
        t_inv_fft_circ=t_inv_fft_circ,
        steps=steps,
        shape=shape,
    ):
        return (
            charge_1
            * charge_2
            * 2
            * jnp.pi
            / math.prod(steps)
            * fft_matvec_solution(t_inv_fft_circ, (d_conj * d).reshape(shape)).ravel()
            * c
        )

    @jax.jit
    def matvec_exchange(
        c,
        d_conj,
        d,
        # NOTE: For the exchange term it does not make sense for there to be
        # two different charges as this term only arises for indistinguishable
        # particles which must have equal charges.
        charge_1=charge_1,
        charge_2=charge_2,
        t_inv_fft_circ=t_inv_fft_circ,
        steps=steps,
        shape=shape,
    ):
        return (
            charge_1
            * charge_2
            * 2
            * jnp.pi
            / math.prod(steps)
            * fft_matvec_solution(t_inv_fft_circ, (d_conj * c).reshape(shape)).ravel()
            * d
        )

    return matvec_direct if kind == "d" else matvec_exchange


@functools.partial(jax.jit, static_argnums=(2, 3))
def setup_t_inv(inds, steps, kinetic_matvec_operator, solver=None):
    shape = [len(ind.ravel()) for ind in inds]
    if solver is None:
        # Use cg by default as the problem is symmetric
        solver = jax.jit(
            jax.scipy.sparse.linalg.cg,
            static_argnums=(0,),
        )

    z = [len(ind.ravel()) // 2 for ind in inds]
    # TODO: Check sharding
    b = sum([jnp.zeros_like(ind).astype(complex) for ind in inds])
    b = b.at[z[0], z[1], z[2]].add(1).ravel()
    A = kinetic_matvec_operator

    return solver(A, b)[0].reshape(shape)


@functools.partial(jax.jit, static_argnums=(2, 3))
def get_t_inv_fft_circ(inds, steps, kinetic_matvec_operator, solver=None):
    return get_fft_embedded_circulant(
        setup_t_inv(inds, steps, kinetic_matvec_operator, solver=solver)
    )


@jax.jit
def setup_t_1d(i, j, step):
    return jnp.where(
        i == j,
        jnp.pi**2 / (6 * step**2),
        (-1.0) ** (i_min_j := i - j) / (step**2 * i_min_j**2),
    )


@jax.jit
def setup_p_1d(i, j, step):
    return (
        -1j
        / step
        * jnp.where(
            i == j,
            0,
            (-1.0) ** (i_min_j := i - j) / i_min_j,
        )
    )


@jax.jit
def get_t_ten(t_vecs):
    assert len(t_vecs) in [1, 2, 3]
    if len(t_vecs) == 1:
        return t_vecs[0]
    deltas = [
        jnp.concatenate([jnp.array([1]), jnp.zeros(len(t_vecs[i]) - 1)])
        for i in range(len(t_vecs))
    ]
    if len(t_vecs) == 2:
        return (
            jnp.kron(t_vecs[0], deltas[1]) + jnp.kron(deltas[0], t_vecs[1])
        ).reshape([len(t) for t in t_vecs])
    return (
        jnp.kron(jnp.kron(t_vecs[0], deltas[1]), deltas[2])
        + jnp.kron(jnp.kron(deltas[0], t_vecs[1]), deltas[2])
        + jnp.kron(jnp.kron(deltas[0], deltas[1]), t_vecs[2])
    ).reshape([len(t) for t in t_vecs])


@jax.jit
def get_t_fft_circ(inds, steps):
    t_vecs = [
        setup_t_1d(ind.ravel(), ind.ravel()[0], step) for ind, step in zip(inds, steps)
    ]
    return get_fft_embedded_circulant(get_t_ten(t_vecs))


@jax.jit
def get_fft_embedded_circulant(t_ten):
    t_slices = [[slice(0, n), 0, slice(n, 0, -1)] for n in t_ten.shape]
    conj_check = [[0, 0, 1] for n in t_ten.shape]
    c_slices = [[slice(0, n), n, slice(n + 1, 2 * n)] for n in t_ten.shape]
    c = jnp.zeros([2 * n for n in t_ten.shape], dtype=t_ten.dtype)
    for c_s, t_s, c_c in zip(
        itertools.product(*c_slices),
        itertools.product(*t_slices),
        itertools.product(*conj_check),
    ):
        if sum(c_c) % 2 == 1:
            c = c.at[c_s].set(t_ten[t_s].conj())
        else:
            c = c.at[c_s].set(t_ten[t_s])
    return jnp.fft.fftn(c)


@jax.jit
def fft_matvec_solution(fft_circ_ten, x_t):
    slices = tuple(slice(0, s) for s in x_t.shape)
    y = jnp.zeros([2 * s for s in x_t.shape], dtype=x_t.dtype)
    y = y.at[slices].set(x_t)
    fft_y = jnp.fft.fftn(y)

    return jnp.fft.ifftn(fft_circ_ten * fft_y)[slices]
