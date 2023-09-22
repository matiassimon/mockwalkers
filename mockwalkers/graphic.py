from abc import ABC, abstractmethod
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
from matplotlib.patches import Patch, Circle, Rectangle
from matplotlib.transforms import IdentityTransform, Affine2D
from matplotlib.backend_bases import RendererBase

from .solver import Solver, Walkers
from .vdcalculator import VdCalculator
from .obstacle import Obstacle, RectangleObstacle


class GraphicElement(ABC):
    @property
    @abstractmethod
    def artists(self):
        pass

    @abstractmethod
    def update(self):
        pass


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


class WalkersElement(GraphicElement):
    def __init__(self, ax: Axes, solver: Solver, **kwargs):
        """"""
        self._ax = ax
        self._solver = solver

        if "color" not in kwargs:
            kwargs["color"] = "C0"

        self._artist = PatchTransCollection(
            (
                Circle(
                    (0, 0),
                    solver.walkers.int_radius,
                ),
            ),
            offsets=solver.walkers.x,
            offset_transform=ax.transData,
            **kwargs
        )

        ax.add_collection(self._artist)
        self._artist.set_transform(IdentityTransform())
        self._artist.set_patch_transforms([ax.transData])

    @property
    def artists(self):
        return [self._artist]

    def update(self):
        self._artist.set_offsets(self._solver.walkers.x)
        return [self._artist]


class ArrowElement(GraphicElement):
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

        self.set_arrows(p, d)

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

    def set_arrows(self, p: ArrayLike, d: ArrayLike):
        self.tails.set_segments(self.__calc_segments(p, d))
        self.heads.set_patch_transforms(self.__calc_heads_transforms(d))
        self.heads.set_offsets(p + d)
        return [self.tails, self.heads]

    @property
    def artists(self):
        return [self.tails, self.heads]


class WalkersFElement(ArrowElement):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(
            ax, solver.walkers.x, solver.f, {"color": "C1"}, {"color": "C1"}
        )

    def update(self):
        return self.set_arrows(self._solver.walkers.x, self._solver.f)


class WalkersKSumElement(ArrowElement):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(
            ax, solver.walkers.x, solver.ksum, {"color": "C2"}, {"color": "C2"}
        )

    def update(self):
        return self.set_arrows(self._solver.walkers.x, self._solver.ksum)


class WalkersEElement(ArrowElement):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        super().__init__(
            ax, solver.walkers.x, solver.e, {"color": "C3"}, {"color": "C3"}
        )

    def update(self):
        return self.set_arrows(self._solver.walkers.x, self._solver.e)


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


class WalkersTracesElement(GraphicElement):
    def __init__(self, ax: Axes, solver: Solver):
        """"""
        self._ax = ax
        self._solver = solver

        linewidths = 2
        zorder = 1
        cmap = plt.get_cmap("turbo_r")
        norm = plt.Normalize(0, 2)
        alpha = 0.5

        self._artist = LineCollection(
            [],
            linewidths=linewidths,
            zorder=zorder,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
        )

        ax.add_collection(self._artist)

        self._segs = WalkersTracesSegments(solver.walkers.n)
        self.update()

    def update(self):
        self._segs.add(self._solver.walkers.x, self._solver.walkers.u)
        self._artist.set_segments(self._segs.segs)
        self._artist.set_array(self._segs.vels)
        return [self._artist]

    @property
    def artists(self):
        return [self._artist]


class SampleAnglesCollection(LineCollection):
    def __init__(self, sample_points: np.ndarray, sampler, angles_scale):
        super().__init__([])
        self._sample_points = sample_points
        self._sampler = sampler
        self._sample_scale = 1
        self._sk = self.__create_sk(0.8 * angles_scale, 0.2 * angles_scale)
        self._nansk = np.empty((3, 2))
        self._nansk.fill(np.nan)

    def __create_sk(self, h, w):
        return np.array([[-w / 2, h / 2], [w / 2, 0], [-w / 2, -h / 2]])

    def __calc_segs(self):
        self._samples = self._sampler(self._sample_points)
        self._samples_norm = np.linalg.norm(self._samples, axis=1) / self._sample_scale
        self._samples_angles = np.arctan2(self._samples[:, 1], self._samples[:, 0])

        n = len(self._sample_points)
        trans = Affine2D()
        self._final_segs = np.empty((n, 3, 2))

        for i in range(n):
            if self._samples_norm[i] == 0.0:
                self._final_segs[i] = self._nansk
                continue

            trans.clear()
            trans.scale(self._samples_norm[i], 1)
            trans.rotate(self._samples_angles[i])
            trans.translate(self._sample_points[i][0], self._sample_points[i][1])
            self._final_segs[i] = trans.transform(self._sk)

        return self._final_segs

    def draw(self, renderer: RendererBase) -> None:
        self.set_segments(self.__calc_segs())
        return super().draw(renderer)


class VdCalcElement(GraphicElement):
    def __init__(self, ax: Axes, vdcalc: VdCalculator):
        self._ax = ax
        self._vdcalc = vdcalc
        n = 50

        sample_points = self.__calc_sample_points(n)

        def sampler(sample_points):
            sample_walkers = Walkers(
                ax.transData.inverted().transform(
                    self._ax.transAxes.transform(sample_points)
                ),
                np.empty(sample_points.shape),
            )
            return vdcalc(sample_walkers)

        self._artist = SampleAnglesCollection(sample_points, sampler, 1 / n)

        ax.add_collection(self._artist)
        self._artist.set_transform(ax.transAxes)
        self._artist.set_color("lightgray")
        self._artist.set_zorder(-2)

    def __calc_sample_points(self, n):
        [l, s] = np.linspace(0, 1, n, endpoint=False, retstep=True)
        l += s / 2
        return np.stack(np.meshgrid(l, l), axis=2).reshape(n * n, 2)

    def update(self):
        return []

    @property
    def artists(self):
        return [self._artist]


class ObstacleElement(GraphicElement):
    def __init__(self, ax: Axes, obstacle: Obstacle):
        self._ax = ax
        self._artist = self.__patch_factory(obstacle)

        ax.add_patch(self._artist)
        self._artist.set_zorder(-1)
        self._artist.set_color("darkgray")

    def __patch_factory(self, obstacle: Obstacle):
        if isinstance(obstacle, RectangleObstacle):
            return Rectangle(obstacle.xy, obstacle.width, obstacle.height)

        raise ValueError("unsoported obstacle")

    @property
    def artists(self):
        return [self._artist]

    def update(self):
        return []


class Graphic(GraphicElement):
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
        f_arrows: bool = False,
        ksum_arrows: bool = False,
        e_arrows: bool = False,
        traces: bool = True,
        vd_calcs: bool = True,
        obstacles: bool = True,
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        self.all_elements = []
        self.update_elements = []
        self.walkers = WalkersElement(ax, solver)
        self.all_elements.append(self.walkers)
        self.update_elements.append(self.walkers)

        if f_arrows:
            self.walkers_f = WalkersFElement(ax, solver)
            self.all_elements.append(self.walkers_f)
            self.update_elements.append(self.walkers_f)

        if ksum_arrows:
            self.walkers_ksum = WalkersKSumElement(ax, solver)
            self.all_elements.append(self.walkers_ksum)
            self.update_elements.append(self.walkers_ksum)

        if e_arrows:
            self.walkers_e = WalkersEElement(ax, solver)
            self.all_elements.append(self.walkers_e)
            self.update_elements.append(self.walkers_e)

        if traces:
            self.walkers_traces = WalkersTracesElement(ax, solver)
            self.all_elements.append(self.walkers_traces)
            self.update_elements.append(self.walkers_traces)

        if vd_calcs:
            self.vd_calcs = [VdCalcElement(ax, vd_calc) for vd_calc in solver.vd_calcs]
            self.all_elements += self.vd_calcs

        if obstacles:
            self.obstacles = [
                ObstacleElement(ax, obstacle) for obstacle in solver.obstacles
            ]
            self.all_elements += self.obstacles

    @property
    def artists(self):
        return [a for element in self.all_elements for a in element.artists]

    @property
    def update_artists(self):
        return [a for element in self.update_elements for a in element.artists]

    def update(self):
        return [a for element in self.update_elements for a in element.update()]

    def remove(self):
        for artist in self.artists:
            artist.remove()
