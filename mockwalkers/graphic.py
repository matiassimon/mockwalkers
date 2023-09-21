import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.collections import (
    LineCollection,
    PatchCollection,
)
from matplotlib.markers import MarkerStyle
from matplotlib.patheffects import Stroke
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.transforms import IdentityTransform, Affine2D
from matplotlib.backend_bases import RendererBase

from .solver import Solver, Walkers
from .vdcalculator import VdCalculator


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


class WalkersCollection(PatchTransCollection):
    def __init__(self, ax: Axes, solver: Solver, **kwargs):
        """"""
        self._ax = ax
        self._solver = solver

        if "color" not in kwargs:
            kwargs["color"] = "C0"

        super().__init__(
            (
                Circle(
                    (0, 0),
                    solver.int_radius,
                ),
            ),
            offsets=solver.walkers.x,
            offset_transform=ax.transData,
            **kwargs
        )

        ax.add_collection(self)
        self.set_transform(IdentityTransform())
        self.set_patch_transforms([ax.transData])

    def update(self):
        self.set_offsets(self._solver.walkers.x)


class ArrowCollections:
    def __init__(
        self,
        ax: Axes,
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
            (MarkerStyle(">"),), offset_transform=self._ax.transData, **kwargs
        )
        # heads.set_patch_transforms([Affine2D().rotate_deg(15)])
        self._ax.add_collection(heads)
        heads.set_transform(IdentityTransform())
        self._base_heads_transforms = Affine2D().scale(marker_scale)
        heads.set_patch_transforms([self._base_heads_transforms])
        return heads

    def __calc_segments(self, p, d):
        return np.stack([p, p + d], axis=1)

    def __calc_heads_transforms(self, d):
        return [
            self._base_heads_transforms + Affine2D().rotate(angle)
            for angle in np.arctan2(d[:, 1], d[:, 0])
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


class WalkersPropulsionCollection(ArrowCollections):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(
            ax, solver.walkers.x, solver.f, {"color": "C1"}, {"color": "C1"}
        )

    def update(self):
        super().update(self._solver.walkers.x, self._solver.f)


class WalkersInteractionCollection(ArrowCollections):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(
            ax, solver.walkers.x, solver.ksum, {"color": "C2"}, {"color": "C2"}
        )

    def update(self):
        super().update(self._solver.walkers.x, self._solver.ksum)


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


class WalkersTracesCollection(LineCollection):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        linewidths = 2
        zorder = 1
        cmap = plt.get_cmap("turbo_r")
        norm = plt.Normalize(0, 2)
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

        self._segs = WalkersTracesSegments(solver.walkers.n)
        self.update()

    def update(self):
        self._segs.add(self._solver.walkers.x, self._solver.walkers.u)
        self.set_segments(self._segs.segs)
        self.set_array(self._segs.vels)


class VdCalculatorLineCollection(LineCollection):
    def __init__(self, ax: Axes, vdcalc: VdCalculator):
        self._vdcalc = vdcalc
        self._ax = ax
        self._n = 50
        self._base_sample_points = self.__calc_base_sample_points(self._n)
        self._vdscale = 1
        self._sk = self.__create_sk(0.8 * (1 / self._n), 0.2 * (1 / self._n))
        self._nansk = np.empty((3, 2))
        self._nansk.fill(np.nan)

        super().__init__([])

        ax.add_collection(self)
        self.set_transform(ax.transAxes)
        self.set_color("lightgray")
        self.set_zorder(-1)

    def __calc_base_sample_points(self, n):
        [l, s] = np.linspace(0, 1, n, endpoint=False, retstep=True)
        l += s / 2
        return np.stack(np.meshgrid(l, l), axis=2).reshape(n * n, 2)

    def __create_sk(self, h, w):
        return np.array([[-w / 2, h / 2], [w / 2, 0], [-w / 2, -h / 2]])

    def __calc_segs(self):
        n = self._n

        self._sample_walkers = Walkers(
            self._ax.transData.inverted().transform(
                self._ax.transAxes.transform(self._base_sample_points)
            ),
            np.empty(self._base_sample_points.shape),
        )
        self._sample_vds = self._vdcalc(self._sample_walkers)
        self._sample_vds_norm = np.linalg.norm(self._sample_vds, axis=1) / self._vdscale
        self._sample_vds_angles = np.arctan2(
            self._sample_vds[:, 1], self._sample_vds[:, 0]
        )

        trans = Affine2D()
        self._final_segs = np.empty((n * n, 3, 2))

        for i in range(n * n):
            if self._sample_vds_norm[i] == 0.0:
                self._final_segs[i] = self._nansk
                continue

            trans.clear()
            trans.scale(self._sample_vds_norm[i], 1)
            trans.rotate(self._sample_vds_angles[i])
            trans.translate(
                self._base_sample_points[i][0], self._base_sample_points[i][1]
            )
            self._final_segs[i] = trans.transform(self._sk)

        return self._final_segs

    def draw(self, renderer: RendererBase) -> None:
        self.set_segments(self.__calc_segs())
        return super().draw(renderer)


class Graphic:
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        self.walkers = WalkersCollection(ax, solver)
        self.walkers_propulsion = WalkersPropulsionCollection(ax, solver)
        self.walkers_interaction = WalkersInteractionCollection(ax, solver)
        self.walkers_traces = WalkersTracesCollection(ax, solver)
        self.vd_calcs = [
            VdCalculatorLineCollection(ax, vd_calc) for vd_calc in solver.vd_calcs
        ]

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
        for vd_calc in self.vd_calcs:
            vd_calc.remove()
