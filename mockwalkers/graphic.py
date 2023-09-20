from matplotlib.artist import allow_rasterization
import numpy as np
from numpy.typing import ArrayLike
import matplotlib as mpl
from matplotlib.collections import (
    LineCollection,
    PatchCollection,
)
from matplotlib.markers import MarkerStyle
from matplotlib.patheffects import Stroke
from mockwalkers.solver import Solver

import matplotlib.transforms as mtransform


class PatchTransCollection(PatchCollection):
    _patch_transforms = None
    _mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    def set_patch_transforms(self, transforms):
        self._patch_transforms = transforms

    def get_transforms(self):
        if self._patch_transforms is None:
            return np.empty((0, 3, 3))

        return [
            np.where(self._mask, trans.get_matrix(), 0)
            for trans in self._patch_transforms
        ]


class Walkers(PatchTransCollection):
    def __init__(self, ax: mpl.axes.Axes, solver: Solver, **kwargs):
        """"""
        self._ax = ax
        self._solver = solver

        if "color" not in kwargs:
            kwargs["color"] = "C0"

        super().__init__(
            (
                mpl.patches.Circle(
                    (0, 0),
                    solver.int_radius,
                ),
            ),
            offsets=solver.x,
            offset_transform=ax.transData,
            **kwargs
        )

        ax.add_collection(self)
        self.set_transform(mtransform.IdentityTransform())
        self.set_patch_transforms([ax.transData])

    def update(self):
        self.set_offsets(self._solver.x)


class ArrowCollections:
    def __init__(
        self,
        ax: mpl.axes.Axes,
        p: ArrayLike,
        d: ArrayLike,
        tails_kwargs: dict = {},
        heads_kwargs: dict = {},
    ):
        self._ax = ax
        p = np.asanyarray(p)
        d = np.asanyarray(d)

        self.tails = self.__create_tails(**tails_kwargs)
        self.heads = self.__create_heads(**heads_kwargs)

        self.__update(p, d)

    def __create_tails(self, **kwargs):
        tails = LineCollection([], **kwargs)
        self._ax.add_collection(tails)
        return tails

    def __create_heads(self, **kwargs):
        marker_scale = kwargs.get("markerscale", 10)

        heads = PatchTransCollection(
            (MarkerStyle("v"),), offset_transform=self._ax.transData, **kwargs
        )
        # heads.set_patch_transforms([mtransform.Affine2D().rotate_deg(15)])
        self._ax.add_collection(heads)
        heads.set_transform(mtransform.IdentityTransform())
        self._base_heads_transforms = mtransform.Affine2D().scale(
            marker_scale
        )
        heads.set_patch_transforms([self._base_heads_transforms])
        return heads

    def __calc_segments(self, p, d):
        return np.stack([p, p + d], axis=1)

    def __calc_heads_transforms(self, d):
        return [
            self._base_heads_transforms + mtransform.Affine2D().rotate(angle)
            for angle in np.arctan2(d[:, 0], d[:, 1])
        ]

    def __update(self, p: ArrayLike, d: ArrayLike):
        self.tails.set_segments(self.__calc_segments(p, d))
        self.heads.set_patch_transforms(self.__calc_heads_transforms(d))
        self.heads.set_offsets(p + d)

    def update(self, p: ArrayLike, d: ArrayLike):
        self.__update(p, d)

    def set_visible(self, b):
        self.tails.set_visible(b)
        self.heads.set_visible(b)

    def remove(self):
        self.tails.remove()
        self.heads.remove()

class WalkersPropulsion(ArrowCollections):
    def __init__(self, ax: mpl.axes.Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(ax, solver.x, solver.f, {"color": "C1"}, {"color": "C1"})

    def update(self):
        super().update(self._solver.x, self._solver.f)


class WalkersInteraction(ArrowCollections):
    def __init__(self, ax: mpl.axes.Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(ax, solver.x, solver.ksum, {"color": "C2"}, {"color": "C2"})

    def update(self):
        super().update(self._solver.x, self._solver.ksum)


class WalkersTracesSegments:
    def __init__(self, nwalkers, size=100):
        self._size = size
        self._nwalkers = nwalkers
        self.segs = np.empty((nwalkers * size, 2, 2))
        self.segs.fill(np.nan)
        self._segs_idx = 0
        self._last_x = np.empty((nwalkers, 2))
        self._last_x.fill(np.nan)
        self.vels = np.empty((nwalkers * size))
        self.vels.fill(np.nan)

    def add(self, x: np.ndarray, u: np.ndarray):
        if x.shape != (self._nwalkers, 2) or u.shape != (self._nwalkers, 2):
            raise ValueError("invalid shape")

        new_segs = np.stack((self._last_x, x), axis=1)
        self.segs[self._segs_idx : self._segs_idx + self._nwalkers] = new_segs

        new_vels = np.linalg.norm(u, axis=1)
        self.vels[self._segs_idx : self._segs_idx + self._nwalkers] = new_vels

        self._last_x = x
        self._segs_idx += self._nwalkers
        self._segs_idx %= self._nwalkers * self._size


class WalkersTraces(LineCollection):
    def __init__(self, ax: mpl.axes.Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        linewidths = 2
        zorder = 1
        cmap = mpl.pyplot.get_cmap("turbo_r")
        norm = mpl.pyplot.Normalize(0, 2)
        path_effects = [Stroke(capstyle="round")]
        alpha = 0.5

        super().__init__(
            [],
            linewidths=linewidths,
            zorder=zorder,
            cmap=cmap,
            norm=norm,
            path_effects=path_effects,
            alpha=alpha,
        )

        ax.add_collection(self)

        self._segs = WalkersTracesSegments(solver.n)
        self.update()

    def update(self):
        self._segs.add(self._solver.x, self._solver.u)
        self.set_segments(self._segs.segs)
        self.set_array(self._segs.vels)


class Graphic:
    def __init__(
        self,
        ax: mpl.axes.Axes,
        solver: Solver,
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        self.walkers = Walkers(ax, solver)
        self.walkers_propulsion = WalkersPropulsion(ax, solver)
        self.walkers_interaction = WalkersInteraction(ax, solver)
        self.walkers_traces = WalkersTraces(ax, solver)

    def update(self):
        self.walkers.update()
        self.walkers_propulsion.update()
        self.walkers_interaction.update()
        self.walkers_traces.update()

    def remove(self):
        self.walkers.remove()
        self.walkers_propulsion.remove()
        self.walkers_interaction.remove()
        self.walkers_traces.remove()
