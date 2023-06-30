from dataclasses import dataclass


class SincDVR:
    def __init__(
        self,
        dim: int,
        num_elements: tuple[int] = None,
        steps: tuple[float] = None,
    ) -> None:
        assert dim in [1, 2, 3]
        assert num_elements is not None or steps is not None


@dataclass
class Quadrature:
    step: float
    num_elements: int
    index_bounds: (int, int)
    grid_bounds: (float, float)
