import math
import itertools
import functools

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
    positive_extent: tuple[float]
        The positive extent of each axis. The full extent for dimension `i`
        will be `extent[i] = [-positive_extent[i], positive_extent[i]]`.
        In order to ensure that we get an integer number of elements, and that
        it splits up evenly on the devices, the program will at times round the
        extent up to the nearest integer.
        This means that the actual extent will always be greater than, or
        equal, to the requested value.
    steps : tuple[float]
        The (uniform) step length in each Cartesian direction. There should
        be one step length for each dimension. Use the same step lengths in
        each direction case of an isotropic grid. The step for a direction is
        also the weight in the quadrature.
    device_shape: tuple[int]
        The distribution of devices on the computer. For a single device in
        three dimensions this is `(1, 1, 1)`. This parameter tells the program
        how the arrays should be sharded.
        Note that the number of devices along an axis must be an odd number to
        ensure that zero is included as a point.
    build_t_inv: bool
        If `True` we solve the Poisson equation in order to build the inverse
        of kinetic energy operator. Otherwise, no inverse is found. This is
        needed for the construction of matrix elements for the Coulomb
        attraction and interaction operators. This flag is only applicable in
        3D, and is ignored for lower dimenisionalities. Note that the number of
        elements must be odd in all directions when using this flag. For some
        values of `positive_extent` (if there is a small extent, I think...)
        the inverse can get a top in the center, for some reason. Default is
        `False`.
    t_inv_solver: function
        A solver for the Poisson equation. Default for `t_inv_solver = None` is
        [jax.scipy.sparse.linalg.cg](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html).
    # n_in_factor: tuple[int]
    #     The number of internal indices (dubbed :math:`n_{small}` in [1]) used
    #     for the solution of the Poisson equation is given by `n_in[i] =
    #     n_in_factor[i] * device_shape[i]`. Ignored if `build_t_inv = False`, or
    #     `num_dim != 3`. For `n_in_factor = None` we use `n_in = element_shape`
    #     (see above). Default is `n_s = None`.
    # n_out_factor: tuple[int]
    #     The number of "far-away"-coordinates used when solving the Poisson
    #     equation (dubbed :math:`n_{big}` in [1]) is `n_out[i] = n_out_factor[i]
    #     * device_shape[i]`. The same conditions as for `n_in_factor` applies.
    #     Note that `n_out_factor[i] >= n_in_factor[i]`. Default is `n_out_factor
    #     = None`.
    verbose: bool
        Toggle used to turn on (`True`) or off (`False`) some output from the
        class. Default is `False`.


    References
    ----------
    [1] J. R. Jones, F. H. Rouet, K. V. Lawler, E. Vecharynski, K. Z. Ibrahim,
    S. Williams, B. Abeln, C. Yang, W. McCurdy, D. J. Haxton, X. S. Li, T. N.
    Rescigno, Molecular Physics, 114, 13, 2014-2018, (2016)
    """

    def __init__(
        self,
        positive_extent: tuple[float],
        steps: tuple[float],
        device_shape: tuple[int],
        build_t_inv: bool = False,
        t_inv_solver=None,
        # n_in_factor: tuple[int] = None,
        # n_out_factor: tuple[int] = None,
        verbose: bool = False,
    ) -> None:
        self.num_dim = len(positive_extent)
        assert self.num_dim in [1, 2, 3]
        assert len(device_shape) == self.num_dim
        assert len(steps) == self.num_dim
        assert all([n_dev % 2 == 1 for n_dev in device_shape])

        num_points_per_device_shape = [
            # Ensure an odd number of points for an odd number of devices
            (_ := math.ceil((math.ceil(pos_ext / dw) * 2 + 1) / n_dev))
            + (_ + n_dev) % 2
            for pos_ext, dw, n_dev in zip(positive_extent, steps, device_shape)
        ]
        num_points_shape = [
            nppds * n_dev
            for nppds, n_dev in zip(num_points_per_device_shape, device_shape)
        ]
        num_points_pos_shape = [(nps - 1) / 2 for nps in num_points_shape]
        new_positive_extent = [
            npps * dw for npps, dw in zip(num_points_pos_shape, steps)
        ]

        if (
            any(
                [
                    (new_pos_ext - pos_ext) != 0
                    for new_pos_ext, pos_ext in zip(
                        new_positive_extent, positive_extent
                    )
                ]
            )
            and verbose
        ):
            print(
                f"Positive extent increased from: {positive_extent} to: {new_positive_extent}"
            )

        self.grid_shape = num_points_shape
        self.steps = steps
        self.device_shape = device_shape

        self.num_elements = math.prod(self.grid_shape)
        self.num_devices = math.prod(self.device_shape)
        self.tot_weight = math.prod(self.steps)

        if verbose:
            print(
                f"Number of elements: {self.num_elements} distributed as: {self.grid_shape}"
            )
            # "Standard" size of a complex number
            c_size = 2 * jnp.zeros(1).dtype.itemsize
            print(
                f"Approximate size requirement for largest tensor (in GB): {2 ** self.num_dim * self.num_elements * c_size / 2 ** 30}"
            )

        assert self.num_elements > 0
        assert self.num_devices > 0
        assert self.tot_weight > 0

        if t_inv_solver is None:
            t_inv_solver = jax.jit(
                jax.scipy.sparse.linalg.cg,
                static_argnums=(0,),
            )

        self.axis_names = [["x", "y", "z"][i] for i in range(self.num_dim)]

        self.mesh = Mesh(
            mesh_utils.create_device_mesh(self.device_shape), axis_names=self.axis_names
        )
        self.spec = P(*self.axis_names)
        self.inds = []

        for i, axis_name in enumerate(self.axis_names):
            self.inds.append(
                (
                    _ := jax.device_put(
                        jnp.arange(self.grid_shape[i]),
                        NamedSharding(self.mesh, P(axis_name)),
                    )
                )
                - max(_) / 2  # Center around zero
            )
            setattr(self, axis_name, self.inds[i] * self.steps[i])
            setattr(self, f"t_{axis_name}", self.setup_t_1d(i))
            setattr(self, f"d_{axis_name}", self.setup_d_1d(i))

        # Kinetic operator matrix elements embedded in a circulant tensor
        # TODO: Test device sharding
        self.t_fft_circ = get_fft_embedded_circulant(
            get_t_ten(
                [getattr(self, f"t_{axis_name}")[:, 0] for axis_name in self.axis_names]
            )
        )

        if build_t_inv and self.num_dim == 3:
            assert all([self.grid_shape[i] % 2 == 1 for i in range(self.num_dim)])
            # n_in = [e * d for e, d in zip(n_in_factor or element_factor, device_shape)]
            # n_out = [
            #     e * d for e, d in zip(n_out_factor or element_factor, device_shape)
            # ]

            # assert all([n_out[i] >= n_in[i] for i in range(self.num_dim)])
            # assert all([n_in[i] <= self.grid_shape[i] for i in range(self.num_dim)])
            # assert all(
            #     [
            #         (n_in[i] % 2) == (self.grid_shape[i] % 2)
            #         for i in range(self.num_dim)
            #     ]
            # )
            # assert all([(n_in[i] % 2) == (n_out[i] % 2) for i in range(self.num_dim)])

            # self.ret_inds = [(_ := jnp.arange(o)) - max(_) / 2 for o in n_in]
            # self.sum_inds = [(_ := jnp.arange(s)) - max(_) / 2 for s in n_out]
            # self.t_inv = jax.device_put(
            #     setup_t_inv(self.inds, self.ret_inds, self.sum_inds, self.steps),
            #     NamedSharding(self.mesh, self.spec),
            # )

            zero_locs = [jnp.argwhere(i == 0) for i in self.inds]
            b = jax.device_put(
                jnp.zeros([len(i) for i in self.inds], dtype=complex),
                NamedSharding(self.mesh, self.spec),
            )
            b = b.at[zero_locs[0], zero_locs[1], zero_locs[2]].add(1).ravel()

            A = self.get_kinetic_matvec_operator()
            # Also known as "v" from Ref. [1]
            self.t_inv = jax.device_put(
                t_inv_solver(A, b)[0].reshape(self.grid_shape),
                NamedSharding(self.mesh, self.spec),
            )
            # TODO: Test device sharding
            self.t_inv_fft_circ = get_fft_embedded_circulant(self.t_inv)

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

    def matvec_kinetic(self, c: jax.typing.ArrayLike) -> jax.Array:
        return fft_matvec_solution(self.t_fft_circ, c.reshape(self.grid_shape)).ravel()

    def evaluate_basis_functions(
        self, position: jax.typing.ArrayLike, r_i: list[jax.typing.ArrayLike]
    ) -> jax.Array:
        dim = len(position)
        assert dim == self.num_dim
        assert dim == len(r_i)

        return math.prod(
            [
                1
                / jnp.sqrt(self.steps[i])
                * jnp.sinc((position[i] - r_i[i]) / self.steps[i])
                for i in range(dim)
            ]
        )

    def construct_r_inv_potentials(
        self, centers: list[jax.typing.ArrayLike], charges: list[float]
    ) -> None:
        # Note: all charges must be sent in at the same time
        # Otherwise, the potentials are overwritten on each call
        # The index of a potential corresponds to the center and charge index.
        # That is, they have the same ordering.
        #
        # This could probably be solved in a much cleaner way...

        assert self.num_dim == 3
        assert len(centers) == len(charges)

        # We have to subtract the edge from the centers
        # This is due to the zero location, i.e., the grid point that equals
        # the value zero, is located in the middle of the grid and hence has an
        # index that is not zero.
        # TODO: Should this be handled another way?
        shift = jnp.array([self.x[0], self.y[0], self.z[0]])

        # TODO: Check sharding
        self.r_inv_potentials = [
            2
            * jnp.pi
            / jnp.sqrt(self.tot_weight)
            * q
            * fft_matvec_solution(
                self.t_inv_fft_circ,
                self.evaluate_basis_functions(
                    c + shift,
                    [
                        self.x[:, None, None],
                        self.y[None, :, None],
                        self.z[None, None, :],
                    ],
                ),
            ).ravel()
            for c, q in zip(centers, charges)
        ]

        return self

    def get_kinetic_matvec_operator(self):
        # Note: This operator expects c to be passed in a single vector.
        # If there are more columns, they should be passed in one at a time
        # from the caller.
        @jax.jit
        def matvec_kinetic(
            c,
            # Ensure closure for the parameters below
            t_fft_circ=self.t_fft_circ,
            grid_shape=self.grid_shape,
        ):
            return fft_matvec_solution(t_fft_circ, c.reshape(grid_shape)).ravel()

        return matvec_kinetic

    def get_coulomb_interaction_matvec_operator(
        self,
        charge_1: float,
        charge_2: float,
        kind: str,
    ):
        """
        Parameters
        ----------
        charge_1 : float
            The charge of the first particle.
        charge_2 : float
            The charge of the second particle.
        kind : str
            Toggle which kind of Coulomb interaction operator.
            Valid values are `"d"` for direct, and `"e"` for exchange.
        """
        assert self.num_dim == 3
        assert kind in ["d", "e"]

        @jax.jit
        def matvec_direct(
            d_conj,
            d,
            c,
            # Ensure closure for the parameters below
            charge_1=charge_1,
            charge_2=charge_2,
            tot_weight=self.tot_weight,
            t_inv_fft_circ=self.t_inv_fft_circ,
            grid_shape=self.grid_shape,
        ):
            return (
                charge_1
                * charge_2
                * 2
                * jnp.pi
                / tot_weight
                * fft_matvec_solution(
                    t_inv_fft_circ, (d_conj * d).reshape(grid_shape)
                ).ravel()
                * c
            )

        @jax.jit
        def matvec_exchange(
            d_conj,
            d,
            c,
            # Ensure closure for the parameters below
            charge_1=charge_1,
            charge_2=charge_2,
            tot_weight=self.tot_weight,
            t_inv_fft_circ=self.t_inv_fft_circ,
            grid_shape=self.grid_shape,
        ):
            return (
                charge_1
                * charge_2
                * 2
                * jnp.pi
                / tot_weight
                * fft_matvec_solution(
                    t_inv_fft_circ, (d_conj * c).reshape(grid_shape)
                ).ravel()
                * d
            )

        return matvec_direct if kind == "d" else matvec_exchange


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
    ret_inds: list[jax.typing.ArrayLike],
    sum_inds: list[jax.typing.ArrayLike],
    steps: list[float],
) -> jax.Array:
    b = poisson_rhs_generalized(ret_inds, sum_inds, steps).ravel()
    A = PoissonLHS(ret_inds, steps)

    x, _ = jax.scipy.sparse.linalg.cg(A, b)

    assert jnp.allclose(A(x), b, atol=1e-5)

    v_inner = x.reshape(tuple(len(o) for o in ret_inds))

    n_out = [max(o) for o in ret_inds]

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

    if (
        len(x_mask) < len(inds[0])
        or len(y_mask) < len(inds[1])
        or len(z_mask) < len(inds[2])
    ):
        assert jnp.sum(jnp.abs(v[jnp.ix_(x_mask, y_mask, z_mask)])) > 1e-12

    return v


def poisson_rhs_generalized(
    ret_inds: tuple[jax.typing.ArrayLike],
    sum_inds: tuple[jax.typing.ArrayLike],
    steps: tuple[float],
) -> jax.Array:
    # Note that ret_inds and sum_inds have to have the same spacing.
    # This means that if one is even, then other must be even too.
    # And, if one contains the zero index, then the other needs to as well.

    t = [
        setup_t_1d(o[:, None], s[None, :], dw)
        for o, s, dw in zip(ret_inds, sum_inds, steps)
    ]
    n_out = [jnp.max(o) for o in ret_inds]

    v_x = v_far_away(
        sum_inds[0][:, None, None],
        ret_inds[1][None, :, None],
        ret_inds[2][None, None, :],
        n_out,
        steps,
    )
    v_y = v_far_away(
        ret_inds[0][:, None, None],
        sum_inds[1][None, :, None],
        ret_inds[2][None, None, :],
        n_out,
        steps,
    )
    v_z = v_far_away(
        ret_inds[0][:, None, None],
        ret_inds[1][None, :, None],
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

    zero_locs = [jnp.argwhere(o == 0) for o in ret_inds]
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
        self, ret_inds: list[jax.typing.ArrayLike], steps: tuple[float]
    ) -> None:
        self.ret_inds = ret_inds
        self.steps = steps

        self.t = [
            setup_t_1d(o[:, None], o[None, :], dw)
            for o, dw in zip(self.ret_inds, self.steps)
        ]
        self.v_shape = [len(s) for s in self.ret_inds]

    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, v: jax.typing.ArrayLike) -> jax.Array:
        v = v.reshape(self.v_shape)

        return (
            jnp.einsum("ip, pjk -> ijk", self.t[0], v)
            + jnp.einsum("jp, ipk -> ijk", self.t[1], v)
            + jnp.einsum("kp, ijp -> ijk", self.t[2], v)
        ).ravel()


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
