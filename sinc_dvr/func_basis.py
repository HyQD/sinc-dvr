import math
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


def get_kinetic_matvec_operator(inds, steps, t_fft_circ=None):
    shape = [len(ind.ravel()) for ind in inds]

    if t_fft_circ is None:
        t_fft_circ = get_t_fft_circ(inds, steps)

    @jax.jit
    def matvec_kinetic(c, t_fft_circ=t_fft_circ, shape=shape):
        return fft_matvec_solution(t_fft_circ, c.reshape(shape)).ravel()

    return matvec_kinetic


def get_r_inv_potential_function(inds, steps, *args, t_inv_fft_circ=None, **kwargs):
    if t_inv_fft_circ is None:
        t_inv_fft_circ = get_t_inv_fft_circ(inds, steps, *args, **kwargs)

    @jax.jit
    def r_inv_potential(
        center, charge, inds=inds, steps=steps, t_inv_fft_circ=t_inv_fft_circ
    ):
        return (
            2
            * jnp.pi
            / jnp.sqrt(math.prod(steps))
            * charge
            * fft_matvec_solution(t_inv_fft_circ, evaluate_spfs(inds, steps, center))
        )

    return r_inv_potential


@jax.jit
def setup_t_inv(inds, steps, solver=None, kinetic_matvec_operator=None):
    if solver is None:
        # Use cg by default as the problem is symmetric
        solver = jax.jit(
            jax.scipy.sparse.linalg.cg,
            static_argnums=(0,),
        )

    if kinetic_matvec_operator is None:
        kinetic_matvec_operator = get_kinetic_matvec_operator(inds, steps)

    z = [jnp.argwhere(i == 0) for i in inds]
    # TODO: Check sharding
    b = sum([jnp.zeros_like(ind).astype(complex) for ind in inds])
    b = b.at[z[0], z[1], z[2]].add(1).ravel()
    A = kinetic_matvec_operator

    return solver(A, b)[0]


@jax.jit
def get_t_inv_fft_circ(*args, t_inv=None, **kwargs):
    if t_inv is None:
        t_inv = setup_t_inv(*args, **kwargs)

    return get_fft_embedded_circulant(t_inv)


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
def get_t_fft_circ(inds, steps):
    t_vecs = [
        setup_t_1d(ind.ravel(), ind.ravel()[0], step) for ind, step in zip(inds, steps)
    ]
    t_ten = get_t_ten(t_vecs)
    return get_fft_embedded_circulant(t_ten)


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
def get_fft_embedded_circulant(t_ten):
    t_slices = [[slice(0, n), 0, slice(n, 0, -1)] for n in t_ten.shape]
    c_slices = [[slice(0, n), n, slice(n + 1, 2 * n)] for n in t_ten.shape]
    c = jnp.zeros([2 * n for n in t_ten.shape], dtype=t_ten.dtype)
    for c_s, t_s in zip(
        itertools.product(*c_slices),
        itertools.product(*t_slices),
    ):
        c = c.at[c_s].set(t_ten[t_s])
    return jnp.fft.fftn(c)


@jax.jit
def fft_matvec_solution(fft_circ_ten, x_t):
    slices = tuple(slice(0, s) for s in x_t.shape)
    y = jnp.zeros([2 * s for s in x_t.shape], dtype=x_t.dtype)
    y = y.at[slices].set(x_t)
    fft_y = jnp.fft.fftn(y)

    return jnp.fft.ifftn(fft_circ_ten * fft_y)[slices]
