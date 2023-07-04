import math
from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.typing import ArrayLike


class SincDVR:
    """
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
    """

    def __init__(
        self,
        num_dim: int,
        steps: tuple[float],
        element_factor: tuple[int],
        device_shape: tuple[int],
    ) -> None:
        assert num_dim in [1, 2, 3]
        assert len(device_shape) == num_dim
        assert len(device_shape) == len(element_factor)
        assert len(device_shape) == len(steps)

        self.num_dim = num_dim

        self.element_shape = (e * d for e, d in zip(element_factor, device_shape))
        self.steps = steps
        self.device_shape = device_shape
        self.num_elements = math.prod(self.element_shape)
        self.num_devices = math.prod(self.device_shape)
        self.tot_weight = math.prod(self.steps)

        assert self.num_elements > 0
        assert self.num_devices > 0
        assert self.tot_weight > 0

        self.axis_names = (["x", "y", "z"][i] for i in range(self.num_dim))

        self.mesh = Mesh(
            mesh_utils.create_device_mesh(self.device_shape), axis_names=self.axis_names
        )
        self.spec = P(*self.axis_names)

        for i, axis_name in enumerate(self.axis_names):
            setattr(self, axis_name, self.setup_grid(i))
            setattr(self, f"t_{axis_name}", self.setup_t_1d(i))
            setattr(self, f"d_{axis_name}", self.setup_d_1d(i))

    def setup_grid(self, axis: int) -> jax.Array:
        assert axis in list(range(self.num_dim))

        step = self.steps[axis]
        num_elements = self.element_shape[axis]

        edge = 0
        if num_elements % 2 == 0:  # Even number of elements
            edge = step * (num_elements // 2 - 0.5)
        else:  # Odd number of elements, zero is included
            edge = step * num_elements // 2

        return jax.device_put(
            jnp.linspace(-edge, edge, num_elements),
            NamedSharding(self.mesh, P(self.axis_names[axis])),
        )

    def setup_t_1d(self, axis: int) -> jax.Array:
        assert axis in list(range(self.num_dim))

        step = self.steps[axis]
        inds = jnp.arange(self.element_shape[axis])

        i = inds[:, None]
        j = inds[None, :]

        return jax.device_put(
            jnp.where(
                i == j,
                jnp.pi**2 / (6 * step**2),
                (-1.0) ** (i_min_j := i - j) / (step**2 * i_min_j**2),
            ),
            NamedSharding(self.mesh, P(self.axis_names[axis])),
        )

    def setup_d_1d(self, axis: int) -> jax.Array:
        assert axis in list(range(self.num_dim))

        step = self.steps[axis]
        inds = jnp.arange(self.element_shape[axis])

        i = inds[:, None]
        j = inds[None, :]

        return jax.device_put(
            (
                1
                / dx
                * jnp.where(
                    i == j,
                    0,
                    (-1.0) ** (i_min_j := i - j) / i_min_j,
                )
            ),
            NamedSharding(self.mesh, P(self.axis_names[axis])),
        )


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
