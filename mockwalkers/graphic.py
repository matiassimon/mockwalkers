import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.collections as collections
import numpy as np
from mockwalkers import solver

CORRIDOR_WALL_THICKNESS = 0.5
GRAPHIC_XLIM = (-1, solver.CORRIDOR_LENGTH + 1)
GRAPHIC_YLIM = (-1, solver.CORRIDOR_WIDTH + 1)
GRAPHIC_TRACES_MAXLEN = 100

class Graphic:
    def __init__(self, ax: plt.Axes, solver: solver.Solver) -> None:
        ax.set_xlim(GRAPHIC_XLIM)
        ax.set_ylim(GRAPHIC_YLIM)
        self._n_walkers = solver.x.shape[0]
        self._obstacles = self.__create_obstacles(ax)
        self._traces_on = False
        self._traces = self.__create_traces(ax)
        self._walkers = self.__create_walkers(ax, solver.types)

        ax.spines['bottom'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray') 
        ax.spines['right'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
        ax.tick_params(axis='x', colors='lightgray')
        ax.tick_params(axis='y', colors='lightgray')
        ax.yaxis.label.set_color('lightgray')
        ax.xaxis.label.set_color('lightgray')

        self.update(solver)

    @property
    def traces_on(self):
        return self._traces_on

    @traces_on.setter
    def traces_on(self, traces_on):
        self._traces_on = traces_on
        if not traces_on:
            self._traces.set_segments([])
        return self._traces_on

    def update(self, solver: solver.Solver) -> None:
        if self._traces_on:
            self.__update_traces(solver.x, solver.u)
        self.__update_walkers(solver.x)

    def __create_obstacles(self, ax: plt.Axes) -> list[plt.Rectangle]:
        bottom_wall = self.__create_wall(ax, (0, -CORRIDOR_WALL_THICKNESS), solver.CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS)
        top_wall = self.__create_wall(ax, (0, solver.CORRIDOR_WIDTH), solver.CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS)
        return [bottom_wall, top_wall]

    def __create_wall(self, ax: plt.Axes, xy: tuple[float], width: float, height: float) -> plt.Rectangle:
        ret = patches.Rectangle(xy, width, height, facecolor="lightgrey")
        ax.add_patch(ret)
        return ret

    def __create_walkers(self, ax: plt.Axes, types: np.ndarray) -> collections.PathCollection:
        return ax.scatter(np.zeros((self._n_walkers, 1)), np.zeros((self._n_walkers, 1)), c=types, zorder = 2)

    def __create_traces(self, ax: plt.Axes) -> collections.PathCollection:
        cmap = plt.get_cmap("turbo_r")
        norm = plt.Normalize(0, 2)
        lc = collections.LineCollection([], zorder = 1, cmap = cmap, norm = norm, path_effects=[path_effects.Stroke(capstyle="round")], alpha=0.25)
        lc.set_linewidth(2)
        ax.add_collection(lc)

        self._trace_segs = np.empty((GRAPHIC_TRACES_MAXLEN, self._n_walkers, 2, 2)) # one segment = 2 points.
        self._trace_segs.fill(np.nan)
        self._trace_segs_idx = 0

        self._last_trace_x = np.empty((self._n_walkers, 2))
        self._last_trace_x.fill(np.nan)

        self._trace_vels = np.empty((GRAPHIC_TRACES_MAXLEN, self._n_walkers)) # norm of the vel.
        self._trace_vels.fill(np.nan)

        return lc

    def __update_walkers(self, x: np.ndarray) -> None:
        if x.shape != (self._n_walkers, 2):
            raise ValueError("invalid shape")
        self._walkers.set_offsets(x)

    def __update_traces(self, x: np.ndarray, u: np.ndarray) -> None:
        if x.shape != (self._n_walkers, 2) or u.shape != (self._n_walkers, 2):
            raise ValueError("invalid shape")

        new_segs = np.stack((self._last_trace_x, x), axis = 1)
        self._trace_segs[self._trace_segs_idx] = new_segs
        self._traces.set_segments(self._trace_segs.reshape((-1,2,2)))

        self._trace_vels[self._trace_segs_idx] = np.linalg.norm(u, axis=1)
        self._traces.set_array(self._trace_vels.reshape(-1))

        self._last_trace_x = x
        self._trace_segs_idx += 1
        self._trace_segs_idx %= GRAPHIC_TRACES_MAXLEN



