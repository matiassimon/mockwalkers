class Obstacle:
    def __init__(self, imp_constant: float):
        self._imp_constant = imp_constant

    @property
    def imp_constant(self) -> float:
        return self._imp_constant


class RectangleObstacle(Obstacle):
    def __init__(
        self, x: float, y: float, dx: float, dy: float, imp_constant: float = 0.1
    ):
        self.x1 = min(x, x + dx)
        self.x2 = max(x, x + dx)
        self.y1 = min(y, y + dy)
        self.y2 = max(y, y + dy)
        super().__init__(imp_constant)

    @property
    def xy(self) -> (float, float):
        return (self.x1, self.y1)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1
