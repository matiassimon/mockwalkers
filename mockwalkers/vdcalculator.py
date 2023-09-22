from abc import ABC, abstractmethod
import numpy as np
from .solver import Walkers


class VdCalculator(ABC):
    @abstractmethod
    def __call__(self, walkers: Walkers) -> np.ndarray:
        """"""
        pass


class RectangleVelocityBooster(VdCalculator):
    def __init__(self, x, y, dx, dy, vdx, vdy, types_mask=0xFFFF):
        """"""
        self._x1 = min(x, x + dx)
        self._x2 = max(x, x + dx)
        self._y1 = min(y, y + dy)
        self._y2 = max(y, y + dy)
        self._vdx = vdx
        self._vdy = vdy
        self._types_mask = types_mask

    def __call__(self, walkers: Walkers) -> np.ndarray:
        x = walkers.x
        types = walkers.types
        mask = (
            (x[:, 0] > self._x1)
            & (x[:, 0] < self._x2)
            & (x[:, 1] > self._y1)
            & (x[:, 1] < self._y2)
        )

        if types is not None:
            mask &= np.bitwise_and(types, self._types_mask) != 0x0000
        mask = np.broadcast_to(mask[:, np.newaxis], x.shape)
        return np.where(mask, np.asarray([self._vdx, self._vdy]), np.zeros(2))
