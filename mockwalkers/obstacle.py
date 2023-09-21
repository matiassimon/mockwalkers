from abc import abstractmethod
import numpy as np
from .walkers import Walkers


class Obstacle:
    def __init__(self, imp_constant):
        self._imp_constant = imp_constant

    @property
    def imp_constant(self):
        return self._imp_constant

    @abstractmethod
    def distance(self, walkers: Walkers) -> np.ndarray:
        pass


class RectangleObstacle(Obstacle):
    def __init__(self, x, y, dx, dy, imp_constant=0.1):
        self._x1 = min(x, x + dx)
        self._x2 = max(x, x + dx)
        self._y1 = min(y, y + dy)
        self._y2 = max(y, y + dy)
        super().__init__(imp_constant)

    def distance(self, walkers: Walkers) -> np.ndarray:
        x = walkers.x

        closest_x = np.clip(x[:, 0], self._x1, self.x2)
        closest_y = np.clip(x[:, 1], self._y1, self._y2)
        return np.column_stack((closest_x, closest_y))
