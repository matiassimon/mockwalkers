from abc import ABC, abstractmethod
import numpy as np
from .walkers import Walkers


class Obstacle(ABC):
    def __init__(self, imp_constant):
        self._imp_constant = imp_constant

    @property
    def imp_constant(self):
        return self._imp_constant


class RectangleObstacle(Obstacle):
    def __init__(self, x, y, dx, dy, imp_constant=0.1):
        self.x1 = min(x, x + dx)
        self.x2 = max(x, x + dx)
        self.y1 = min(y, y + dy)
        self.y2 = max(y, y + dy)
        super().__init__(imp_constant)

    @property
    def xy(self):
        return (self.x1, self.y1)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1
