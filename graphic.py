import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import numpy as np
import solver

CORRIDOR_WALL_THICKNESS = 0.5
GRAPHIC_XLIM = (-1, solver.CORRIDOR_LENGTH + 1)
GRAPHIC_YLIM = (-1, solver.CORRIDOR_WIDTH +1)

class Graphic:
    def __init__(self, ax: plt.Axes, solver: solver.Solver) -> None:
        ax.set_xlim(GRAPHIC_XLIM)
        ax.set_ylim(GRAPHIC_YLIM)
        self._obstacles = self.__create_obstacles(ax)
        self._walkers = self.__create_walkers(ax, solver.types)
        self.update(solver)

    def update(self, solver: solver.Solver) -> None:
        self.__update_walkers(solver.x)

    def __create_obstacles(self, ax: plt.Axes) -> list[plt.Rectangle]:
        bottom_wall = self.__create_wall(
            ax, (0, -CORRIDOR_WALL_THICKNESS), solver.CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS
        )
        top_wall = self.__create_wall(
            ax, (0, solver.CORRIDOR_WIDTH), solver.CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS
        )
        return [bottom_wall, top_wall]

    def __create_wall(
        self, ax: plt.Axes, xy: tuple[float], width: float, height: float
    ) -> plt.Rectangle:
        ret = patches.Rectangle(xy, width, height)
        ax.add_patch(ret)
        return ret

    def __create_walkers(
        self, ax: plt.Axes, types: np.ndarray
    ) -> collections.PathCollection:
        return ax.scatter(np.zeros(types.shape), np.zeros(types.shape))

    def __update_walkers(self, x: np.ndarray) -> None:
        self._walkers.set_offsets(x)
