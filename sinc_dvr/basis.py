import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.typing

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


class SincDVR:
    r"""
    Note that the grid is given as vectors. If you want to use a meshgrid,
    e.g., when plotting, then the indexing should be set to "ij" in the
    meshgrid-method (Jax or NumPy).

    Parameters
    ----------
    num_dim : int
        The number of Cartesian dimensions in the problem. Accepted values
        are 1, 2 and 3.
    steps : tuple[float]
        The (uniform) step length in each Cartesian direction. There should
        be one step length for each dimension. Use three of the same step
        lengths in case of an isotropic grid. The step for a direction is also
        the weight in the quadrature.
    element_factor : tuple[int]
        The number of elements along dimension `i` is `element_shape[i] =
        element_factor[i] * device_shape[i]`. This is to ensure that the
        element shape and device shape are congruent. Notably, this means that
        the point `0` in an axis is included only for an odd number of devices
        along that same axis.
    device_shape: tuple[int]
        The distribution of devices on the computer. For a single device in
        three dimensions this is `(1, 1, 1)`. This parameter tells the program
        how the arrays should be sharded.
    build_t_inv: bool
        If `True` we solve the Poisson equation in order to build the inverse
        of kinetic energy operator. Otherwise, no inverse is found. This is
        needed for the construction of matrix elements for the Coulomb
        attraction and interaction operators. This flag is only applicable in
        3D, and is ignored for lower dimenisionalities. Default is `False`.
    n_s: tuple[int]
        The number of internal indices (dubbed :math:`n_{small}` in [1]) used
        for the solution of the Poisson equation. Ignored if `build_t_inv =
        False`, or `num_dim != 3`. For `n_s = None` we use `n_s =
        element_shape` (see above). Default is `n_s = None`.
    n_b: tuple[int]
        The number of "far-away"-coordinates used when solving the Poisson
        equation (dubbed :math:`n_{big}` in [1]). The same conditions as for
        `n_s` applies. Note that `n_b >= n_s`. Default is `n_s = None`.


    References
    ----------
    [1] J. R. Jones, F. H. Rouet, K. V. Lawler, E. Vecharynski, K. Z. Ibrahim,
    S. Williams, B. Abeln, C. Yang, W. McCurdy, D. J. Haxton, X. S. Li, T. N.
    Rescigno, Molecular Physics, 114, 13, 2014-2018, (2016)
    """

    def __init__(
        self,
        num_dim: int,
        steps: tuple[float],
        element_factor: tuple[int],
        device_shape: tuple[int],
        build_t_inv: bool = False,
        n_s: tuple[int] = None,
        n_b: tuple[int] = None,
    ) -> None:
        assert num_dim in [1, 2, 3]
        assert len(device_shape) == num_dim
        assert len(device_shape) == len(element_factor)
        assert len(device_shape) == len(steps)

        self.num_dim = num_dim

        self.element_shape = [e * d for e, d in zip(element_factor, device_shape)]
        self.steps = steps
        self.device_shape = device_shape
        self.num_elements = math.prod(self.element_shape)
        self.num_devices = math.prod(self.device_shape)
        self.tot_weight = math.prod(self.steps)

        assert self.num_elements > 0
        assert self.num_devices > 0
        assert self.tot_weight > 0

        self.axis_names = [["x", "y", "z"][i] for i in range(self.num_dim)]

        self.mesh = Mesh(
            mesh_utils.create_device_mesh(self.device_shape), axis_names=self.axis_names
        )
        self.spec = P(*self.axis_names)
        self.inds = []

        for i, axis_name in enumerate(self.axis_names):
            # setattr(self, axis_name, self.setup_grid(i))
            self.inds.append(
                (
                    _ := jax.device_put(
                        jnp.arange(self.element_shape[i]),
                        NamedSharding(self.mesh, P(axis_name)),
                    )
                )
                - max(_) / 2  # Center around zero
            )
            setattr(self, axis_name, self.inds[i] * self.steps[i])
            setattr(self, f"t_{axis_name}", self.setup_t_1d(i))
            setattr(self, f"d_{axis_name}", self.setup_d_1d(i))

        if build_t_inv and self.num_dim == 3:
            n_s = n_s or self.element_shape
            n_b = n_b or self.element_shape

            assert all([n_b[i] >= n_s[i] for i in range(self.num_dim)])
            assert all([n_s[i] <= self.element_shape[i] for i in range(self.num_dim)])

            self.out_inds = [(_ := jnp.arange(o)) - max(_) / 2 for o in n_s]
            self.sum_inds = [(_ := jnp.arange(s)) - max(_) / 2 for s in n_b]

            # Also known as "v" from Ref. [1]
            self.t_inv = setup_t_inv(
                self.inds, self.out_inds, self.sum_inds, self.steps
            )

        # Kinetic operator matrix elements embedded in a circulant tensor
        self.tc = self.create_kinetic_operator_circulant_tensor()

    def setup_t_1d(self, axis: int) -> jax.Array:
        assert axis in list(range(self.num_dim))

        return jax.device_put(
            setup_t_1d(
                self.inds[axis][:, None],
                self.inds[axis][None, :],
                self.steps[axis],
            ),
            NamedSharding(self.mesh, P(self.axis_names[axis])),
        )

    def setup_d_1d(self, axis: int) -> jax.Array:
        assert axis in list(range(self.num_dim))

        return jax.device_put(
            setup_d_1d(
                self.inds[axis][:, None],
                self.inds[axis][None, :],
                self.steps[axis],
            ),
            NamedSharding(self.mesh, P(self.axis_names[axis])),
        )

    def place_r_inv_charges(
        self, centers: list[jax.typing.ArrayLike], charges: list[jax.typing.ArrayLike]
    ) -> None:
        # Note: all charges must be sent in at the same time
        # Otherwise, the potentials are overwritten on each call
        # The index of a potential corresponds to the center and charge index.
        # That is, they have the same ordering.
        #
        # This could probably be solved in a much cleaner way...

        assert len(centers) == len(charges)

        self.r_inv_potentials = []

    def create_kinetic_operator_circulant_tensor(self):
        # In case of error, check starting point
        t_vecs = [
            getattr(self, f"t_{axis_name}")[:, 0] for axis_name in self.axis_names
        ]
        tc_vecs = [
            jax.device_put(
                jnp.concatenate(
                    [
                        t_vec,
                        jnp.concatenate(
                            [
                                jnp.array([t_vec[0]]),
                                t_vec[1:][::-1],
                            ]
                        ),
                    ]
                ),
                NamedSharding(self.mesh, P(axis_name)),
            )
            for axis_name, t_vec in zip(self.axis_names, t_vecs)
        ]

        if self.num_dim == 1:
            return tc_vecs[0]
        elif self.num_dim == 2:
            # In case of error, check deltas
            # TODO: Check if the deltas should be sharded
            delta_x = jnp.concatenate(
                [jnp.array([1.0]), jnp.zeros(len(tc_vecs[0]) - 1)]
            )
            delta_y = jnp.concatenate(
                [jnp.array([1.0]), jnp.zeros(len(tc_vecs[1]) - 1)]
            )

            # TODO: Check sharding after kronecker products
            return (
                jnp.kron(tc_vecs[0], delta_y) + jnp.kron(delta_x, tc_vecs[1])
            ).reshape((len(tc_vecs[0]), len(tc_vecs[1])))
        else:
            # In case of error, check deltas
            # TODO: Check if the deltas should be sharded
            delta_x = jnp.concatenate(
                [jnp.array([1.0]), jnp.zeros(len(tc_vecs[0]) - 1)]
            )
            delta_y = jnp.concatenate(
                [jnp.array([1.0]), jnp.zeros(len(tc_vecs[1]) - 1)]
            )
            delta_z = jnp.concatenate(
                [jnp.array([1.0]), jnp.zeros(len(tc_vecs[2]) - 1)]
            )

            # TODO: Check sharding after kronecker products
            return (
                jnp.kron(jnp.kron(tc_vecs[0], delta_y), delta_z)
                + jnp.kron(jnp.kron(delta_x, tc_vecs[1]), delta_z)
                + jnp.kron(jnp.kron(delta_x, delta_y), tc_vecs[2])
            ).reshape((len(tc_vecs[0]), len(tc_vecs[1]), len(tc_vecs[2])))

    def matvec_kinetic(self, c: jax.typing.ArrayLike) -> jax.Array:
        c = c.reshape(self.element_shape)
        # Embedding vector
        y = jnp.zeros([e * 2 for e in self.element_shape], dtype=c.dtype)
        y = y.at[: c.shape[0], : c.shape[1], : c.shape[2]].set(c)

        return jnp.fft.ifftn(jnp.fft.fftn(self.tc) * jnp.fft.fftn(y))[
            : c.shape[0], : c.shape[1], : c.shape[2]
        ].ravel()


@jax.jit
def setup_t_1d(
    i: jax.typing.ArrayLike, j: jax.typing.ArrayLike, step: float
) -> jax.Array:
    return jnp.where(
        i == j,
        jnp.pi**2 / (6 * step**2),
        (-1.0) ** (i_min_j := i - j) / (step**2 * i_min_j**2),
    )


@jax.jit
def setup_d_1d(
    i: jax.typing.ArrayLike, j: jax.typing.ArrayLike, step: float
) -> jax.Array:
    return (
        1
        / step
        * jnp.where(
            i == j,
            0,
            (-1.0) ** (i_min_j := i - j) / i_min_j,
        )
    )


def setup_t_inv(
    inds: list[jax.typing.ArrayLike],
    out_inds: list[jax.typing.ArrayLike],
    sum_inds: list[jax.typing.ArrayLike],
    steps: list[float],
) -> jax.Array:
    b = poisson_rhs_generalized(out_inds, sum_inds, steps).ravel()
    A = PoissonLHS(out_inds, steps)

    x, _ = jax.scipy.sparse.linalg.cg(A, b)

    assert jnp.allclose(A(x), b, atol=1e-5)

    v_inner = x.reshape(tuple(len(o) for o in out_inds))

    n_out = [max(o) for o in out_inds]

    v = v_far_away(
        inds[0][:, None, None],
        inds[1][None, :, None],
        inds[2][None, None, :],
        n_out,
        steps,
    )

    x_mask = jnp.arange(len(inds[0]))[abs(inds[0]) <= n_out[0]]
    y_mask = jnp.arange(len(inds[1]))[abs(inds[1]) <= n_out[1]]
    z_mask = jnp.arange(len(inds[2]))[abs(inds[2]) <= n_out[2]]

    assert jnp.sum(jnp.abs(v[jnp.ix_(x_mask, y_mask, z_mask)])) < 1e-12

    v = v.at[jnp.ix_(x_mask, y_mask, z_mask)].set(v_inner)

    assert jnp.sum(jnp.abs(v[jnp.ix_(x_mask, y_mask, z_mask)])) > 1e-12

    return v


def poisson_rhs_generalized(
    out_inds: tuple[jax.typing.ArrayLike],
    sum_inds: tuple[jax.typing.ArrayLike],
    steps: tuple[float],
) -> jax.Array:
    # Note that out_inds and sum_inds have to have the same spacing.
    # This means that if one is even, then other must be even too.
    # And, if one contains the zero index, then the other needs to as well.

    t = [
        setup_t_1d(o[:, None], s[None, :], dw)
        for o, s, dw in zip(out_inds, sum_inds, steps)
    ]
    n_out = [jnp.max(o) for o in out_inds]

    v_x = v_far_away(
        sum_inds[0][:, None, None],
        out_inds[1][None, :, None],
        out_inds[2][None, None, :],
        n_out,
        steps,
    )
    v_y = v_far_away(
        out_inds[0][:, None, None],
        sum_inds[1][None, :, None],
        out_inds[2][None, None, :],
        n_out,
        steps,
    )
    v_z = v_far_away(
        out_inds[0][:, None, None],
        out_inds[1][None, :, None],
        sum_inds[2][None, None, :],
        n_out,
        steps,
    )

    b = (
        -jnp.einsum("ip, pjk -> ijk", t[0], v_x)
        - jnp.einsum("jp, ipk -> ijk", t[1], v_y)
        - jnp.einsum("kp, ijp -> ijk", t[2], v_z)
    )

    n_sum = [jnp.max(s) for s in sum_inds]
    assert all(s >= o for s, o in zip(n_sum, n_out))
    if all(abs(s - o) < 1e-12 for s, o in zip(n_sum, n_out)):
        assert jnp.sum(jnp.abs(b)) < 1e-12

    zero_locs = [jnp.argwhere(o == 0) for o in out_inds]
    b = b.at[zero_locs[0], zero_locs[1], zero_locs[2]].add(1)

    return b


# Jit, in case this should be called in a dynamic setting
@jax.jit
def v_far_away(
    i_1: jax.typing.ArrayLike,
    i_2: jax.typing.ArrayLike,
    i_3: jax.typing.ArrayLike,
    n_s: tuple[float],
    steps: tuple[float],
) -> jax.Array:
    return jnp.where(
        (jnp.abs(i_1) > n_s[0]) | (jnp.abs(i_2) > n_s[1]) | (jnp.abs(i_3) > n_s[2]),
        math.prod(steps)
        / (2 * jnp.pi)
        / jnp.sqrt(
            (i_1 * steps[0]) ** 2 + (i_2 * steps[1]) ** 2 + (i_3 * steps[2]) ** 2
        ),
        0,
    )


class PoissonLHS:
    def __init__(
        self, out_inds: list[jax.typing.ArrayLike], steps: tuple[float]
    ) -> None:
        self.out_inds = out_inds
        self.steps = steps

        self.t = [
            setup_t_1d(o[:, None], o[None, :], dw)
            for o, dw in zip(self.out_inds, self.steps)
        ]
        self.v_shape = [len(s) for s in self.out_inds]

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, v: jax.typing.ArrayLike) -> jax.Array:
        v = v.reshape(self.v_shape)

        return (
            jnp.einsum("ip, pjk -> ijk", self.t[0], v)
            + jnp.einsum("jp, ipk -> ijk", self.t[1], v)
            + jnp.einsum("kp, ijp -> ijk", self.t[2], v)
        ).ravel()


def get_matvec_ein_str(dim):
    assert dim in [1, 2, 3]
    ein_str = ""
    if dim == 1:  # x-axis
        ein_str = "ip, pjk -> ijk"
    elif dim == 2:  # y-axis
        ein_str = "jp, ipk -> ijk"
    elif dim == 3:  # z-axis
        ein_str = "kp, ijp -> ijk"
    else:
        raise NotImplementedError(
            f"Invalid value for dim ({dim}), should be either 1, 2 or 3"
        )

    return ein_str


def get_vecvec_ein_str(dim):
    assert dim in [1, 2, 3]
    ein_str = ""
    if dim == 1:  # x-axis
        ein_str = "i, ijk -> ijk"
    elif dim == 2:  # y-axis
        ein_str = "j, ijk -> ijk"
    elif dim == 3:  # z-axis
        ein_str = "k, ijk -> ijk"
    else:
        raise NotImplementedError(
            f"Invalid value for dim ({dim}), should be either 1, 2 or 3"
        )

    return ein_str
