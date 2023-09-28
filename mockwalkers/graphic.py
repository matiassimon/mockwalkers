from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.collections import (
    LineCollection,
    PatchCollection,
)
from matplotlib.markers import MarkerStyle
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import (
    Affine2DBase,
    AffineDeltaTransform,
    IdentityTransform,
    Affine2D,
)
from matplotlib.backend_bases import RendererBase
from matplotlib.path import Path
from matplotlib.artist import Artist

from .solver import Solver, Walkers
from .vdcalculator import VdCalculator
from .obstacle import Obstacle, RectangleObstacle
from .geometry import Euclidean


class GraphicElement(ABC):
    @property
    @abstractmethod
    def artists(self) -> [Artist]:
        pass

    @abstractmethod
    def update(self) -> [Artist]:
        pass


class PatchTransCollection(PatchCollection):
    _patch_transforms = None

    def set_patch_transforms(self, transforms: [Affine2DBase]) -> None:
        self._patch_transforms = transforms

    def get_transforms(self) -> [np.ndarray]:
        if self._patch_transforms is None:
            return np.empty((0, 3, 3))

        return [trans.get_matrix() for trans in self._patch_transforms]


class WalkersElement(GraphicElement):
    def __init__(self, ax: Axes, solver: Solver, **kwargs) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        if "color" not in kwargs:
            kwargs["color"] = "C0"

        self._artist = PatchTransCollection(
            (
                Circle(
                    (0, 0),
                    solver.walkers.int_radius / 4,
                ),
            ),
            offsets=solver.walkers.x,
            offset_transform=ax.transData,
            **kwargs,
        )

        ax.add_collection(self._artist)
        self._artist.set_transform(IdentityTransform())
        self._artist.set_patch_transforms([AffineDeltaTransform(ax.transData)])

    @property
    def artists(self) -> [Artist]:
        return [self._artist]

    def update(self):
        self._artist.set_offsets(self._solver.walkers.x)
        return [self._artist]


class ArrowHeadTransform(Affine2DBase):
    def __init__(self, d: ArrayLike, scale: float, trans_data: Affine2D) -> None:
        super().__init__()
        self._t = Affine2D()
        self._d = d
        self._scale = scale
        self._trans_data = AffineDeltaTransform(trans_data)
        self.set_children(trans_data)
        self._mtx = None

    @property
    def depth(self) -> int:
        return self._t.depth + self._trans_data.depth

    def get_matrix(self) -> np.ndarray:
        if self._invalid:
            dd = self._trans_data.transform(self._d)
            self._t.clear()
            self._t.rotate(np.arctan2(dd[1], dd[0]))
            self._t.scale(self._scale)
            self._mtx = self._t.get_matrix()
            self._inverted = None
            self._invalid = 0

        return self._mtx


class ArrowElement(GraphicElement):
    def __init__(
        self,
        ax: Axes,
        p: ArrayLike,
        d: ArrayLike,
        *,
        tails_kwargs: dict = {},
        heads_kwargs: dict = {},
    ) -> None:
        self._ax = ax
        p = np.asanyarray(p)
        d = np.asanyarray(d)

        self.tails = self.__create_tails(**tails_kwargs)
        self.heads = self.__create_heads(**heads_kwargs)

        self.set_arrows(p, d)

    def __create_tails(self, **kwargs) -> LineCollection:
        tails = LineCollection([], **kwargs)
        self._ax.add_collection(tails)
        return tails

    def __create_heads(self, **kwargs) -> PatchTransCollection:
        self._head_markerscale = kwargs.get("markerscale", 10)

        heads = PatchTransCollection(
            (MarkerStyle(">"),), offset_transform=self._ax.transData, **kwargs
        )
        self._ax.add_collection(heads)
        heads.set_transform(IdentityTransform())
        return heads

    def __calc_segments(self, p: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.stack([p, p + d], axis=1)

    def __calc_heads_transforms(self, d) -> [ArrowHeadTransform]:
        return [
            ArrowHeadTransform(di, self._head_markerscale, self._ax.transData)
            for di in d
        ]

    def set_arrows(self, p: np.ndarray, d: np.ndarray) -> [Artist]:
        self.tails.set_segments(self.__calc_segments(p, d))
        self.heads.set_patch_transforms(self.__calc_heads_transforms(d))
        self.heads.set_offsets(p + d)
        return [self.tails, self.heads]

    @property
    def artists(self) -> [Artist]:
        return [self.tails, self.heads]


class WalkersFElement(ArrowElement):
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
        *,
        tails_kwargs: dict = {},
        heads_kwargs: dict = {},
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        for kwargs in [tails_kwargs, heads_kwargs]:
            if "color" not in kwargs:
                kwargs["color"] = "C1"

        super().__init__(
            ax,
            solver.walkers.x,
            solver.f,
            tails_kwargs=tails_kwargs,
            heads_kwargs=heads_kwargs,
        )

    def update(self) -> [Artist]:
        return self.set_arrows(self._solver.walkers.x, self._solver.f)


class WalkersKSumElement(ArrowElement):
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
        *,
        tails_kwargs: dict = {},
        heads_kwargs: dict = {},
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        for kwargs in [tails_kwargs, heads_kwargs]:
            if "color" not in kwargs:
                kwargs["color"] = "C2"

        super().__init__(
            ax,
            solver.walkers.x,
            solver.ksum,
            tails_kwargs=tails_kwargs,
            heads_kwargs=heads_kwargs,
        )

    def update(self) -> [Artist]:
        return self.set_arrows(self._solver.walkers.x, self._solver.ksum)


class WalkersEElement(ArrowElement):
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
        *,
        tails_kwargs: dict = {},
        heads_kwargs: dict = {},
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        for kwargs in [tails_kwargs, heads_kwargs]:
            if "color" not in kwargs:
                kwargs["color"] = "C3"

        super().__init__(
            ax,
            solver.walkers.x,
            solver.e,
            tails_kwargs=tails_kwargs,
            heads_kwargs=heads_kwargs,
        )

    def update(self) -> [Artist]:
        return self.set_arrows(self._solver.walkers.x, self._solver.e)


class WalkersTracesLineCollection(LineCollection):
    def __init__(self, nwalkers: int, *, size: int = 100, **kwargs) -> None:
        super().__init__([], **kwargs)

        self._size = size
        self._nwalkers = nwalkers
        self._segs = [Path([[np.nan, np.nan]])] * (nwalkers * size)
        self._segs_idx = 0
        self._last_x = np.empty((nwalkers, 2))
        self._last_x.fill(np.nan)
        self._vels = np.empty((nwalkers * size))
        self._vels.fill(np.nan)

    def add(self, x: np.ndarray, u: np.ndarray) -> None:
        if x.shape != (self._nwalkers, 2) or u.shape != (self._nwalkers, 2):
            raise ValueError("invalid shape")

        new_segs = np.stack((self._last_x, x), axis=1)

        for i in range(self._nwalkers):
            self._segs[self._segs_idx + i] = Path(new_segs[i])

        new_vels = np.linalg.norm(u, axis=1)
        self._vels[self._segs_idx : self._segs_idx + self._nwalkers] = new_vels

        self._last_x = x
        self._segs_idx += self._nwalkers
        self._segs_idx %= self._nwalkers * self._size

        self.set_array(self._vels)
        self.stale = True

    def get_paths(self) -> [Path]:
        return self._segs


class WalkersTracesElement(GraphicElement):
    def __init__(self, ax: Axes, solver: Solver, **kwargs) -> None:
        """"""
        self._ax = ax
        self._solver = solver
        self._skip = kwargs.get("skip", 0)
        self._skipped = 0

        if "linewidths" not in kwargs:
            kwargs["linewidths"] = 2
        if "zorder" not in kwargs:
            kwargs["zorder"] = -1
        if "cmap" not in kwargs:
            kwargs["cmap"] = plt.get_cmap("turbo_r")
        if "norm" not in kwargs:
            kwargs["norm"] = plt.Normalize(0, 2)
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5

        self._artist = WalkersTracesLineCollection(solver.walkers.n, **kwargs)

        ax.add_collection(self._artist)
        self.update()

    def update(self) -> [Artist]:
        if self._skipped != self._skip:
            self._skipped += 1
            return []
        self._skipped = 0

        mask = np.isnan(self._artist._last_x)
        mask |= np.equal(
            Euclidean().distance(self._artist._last_x, self._solver.walkers.x),
            self._solver.geometry.distance(
                self._artist._last_x, self._solver.walkers.x
            ),
        )
        x = np.where(mask, self._solver.walkers.x, np.nan)
        self._artist.add(x, self._solver.walkers.u)
        return [self._artist]

    @property
    def artists(self) -> None:
        return [self._artist]


class SampleAnglesCollection(LineCollection):
    def __init__(
        self,
        sample_points: np.ndarray,
        sampler: Callable[[np.ndarray], np.ndarray],
        *,
        angles_scale,
        **kwargs,
    ) -> None:
        super().__init__([], **kwargs)
        self._sample_points = sample_points
        self._sampler = sampler
        self._sample_scale = 1
        self._sk = self.__create_sk(0.8 * angles_scale, 0.2 * angles_scale)
        self._nansk = np.empty((3, 2))
        self._nansk.fill(np.nan)

    def __create_sk(self, h: float, w: float) -> np.ndarray:
        return np.array([[-w / 2, h / 2], [w / 2, 0], [-w / 2, -h / 2]])

    def __calc_segs(self) -> np.ndarray:
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
    def __init__(self, ax: Axes, vd_calc: VdCalculator, **kwargs):
        self._ax = ax
        self._vd_calc = vd_calc
        n = 50

        sample_points = self.__calc_sample_points(n)

        def sampler(sample_points):
            transDataToAxes = ax.transData + ax.transAxes.inverted()
            deltaTransDataToAxes = AffineDeltaTransform(transDataToAxes)
            sample_walkers = Walkers(
                transDataToAxes.inverted().transform(sample_points),
                np.zeros(sample_points.shape),
            )
            return deltaTransDataToAxes.transform(vd_calc(sample_walkers))

        if "angles_scale" not in kwargs:
            kwargs["angles_scale"] = 1 / n
        if "linewidths" not in kwargs:
            kwargs["linewidths"] = 2
        if "zorder" not in kwargs:
            kwargs["zorder"] = -2
        if "color" not in kwargs:
            kwargs["color"] = "lightgray"

        self._artist = SampleAnglesCollection(sample_points, sampler, **kwargs)
        ax.add_collection(self._artist)
        self._artist.set_transform(ax.transAxes)

    def __calc_sample_points(self, n: int) -> np.ndarray:
        [l, s] = np.linspace(0, 1, n, endpoint=False, retstep=True)
        l += s / 2
        return np.stack(np.meshgrid(l, l), axis=2).reshape(n * n, 2)

    def update(self) -> [Artist]:
        return []

    @property
    def artists(self) -> [Artist]:
        return [self._artist]


class ObstacleElement(GraphicElement):
    def __init__(self, ax: Axes, obstacle: Obstacle, **kwargs):
        self._ax = ax

        if "zorder" not in kwargs:
            kwargs["zorder"] = -1
        if "color" not in kwargs:
            kwargs["color"] = "darkgray"

        self._artist = self.__patch_factory(obstacle, **kwargs)
        ax.add_patch(self._artist)

    def __patch_factory(self, obstacle: Obstacle, **kwargs) -> Artist:
        if isinstance(obstacle, RectangleObstacle):
            return Rectangle(obstacle.xy, obstacle.width, obstacle.height, **kwargs)

        raise ValueError("unsoported obstacle")

    @property
    def artists(self) -> [Artist]:
        return [self._artist]

    def update(self) -> [Artist]:
        return []


class Graphic(GraphicElement):
    def __init__(
        self,
        ax: Axes,
        solver: Solver,
        *,
        walkers_kwargs: dict = {},
        f_arrows: bool = False,
        f_arrows_kwargs: dict = {},
        ksum_arrows: bool = False,
        ksum_arrows_kwargs: dict = {},
        e_arrows: bool = False,
        e_arrows_arrows_kwargs: dict = {},
        traces: bool = True,
        traces_kwargs: dict = {},
        vd_calcs: bool = True,
        vd_calcs_kwargs: [dict] = None,
        obstacles: bool = True,
        obstacles_kwargs: [dict] = None,
    ) -> None:
        """"""
        self._ax = ax
        self._solver = solver

        self.all_elements = []
        self.update_elements = []
        self.walkers = WalkersElement(ax, solver, **walkers_kwargs)
        self.all_elements.append(self.walkers)
        self.update_elements.append(self.walkers)

        if f_arrows:
            self.walkers_f = WalkersFElement(ax, solver, **f_arrows_kwargs)
            self.all_elements.append(self.walkers_f)
            self.update_elements.append(self.walkers_f)

        if ksum_arrows:
            self.walkers_ksum = WalkersKSumElement(ax, solver, **ksum_arrows_kwargs)
            self.all_elements.append(self.walkers_ksum)
            self.update_elements.append(self.walkers_ksum)

        if e_arrows:
            self.walkers_e = WalkersEElement(ax, solver, **e_arrows_arrows_kwargs)
            self.all_elements.append(self.walkers_e)
            self.update_elements.append(self.walkers_e)

        if traces:
            self.walkers_traces = WalkersTracesElement(ax, solver, **traces_kwargs)
            self.all_elements.append(self.walkers_traces)
            self.update_elements.append(self.walkers_traces)

        if vd_calcs:
            if vd_calcs_kwargs is None:
                vd_calcs_kwargs = [{}] * len(solver.vd_calcs)
            self.vd_calcs = [
                VdCalcElement(ax, vd_calc, **vd_calc_kwargs)
                for (vd_calc, vd_calc_kwargs) in zip(
                    solver.vd_calcs, vd_calcs_kwargs, strict=True
                )
            ]
            self.all_elements += self.vd_calcs

        if obstacles:
            if obstacles_kwargs is None:
                obstacles_kwargs = [{}] * len(solver.obstacles)
            self.obstacles = [
                ObstacleElement(ax, obstacle, **obstacle_kwargs)
                for (obstacle, obstacle_kwargs) in zip(
                    solver.obstacles, obstacles_kwargs, strict=True
                )
            ]
            self.all_elements += self.obstacles

    @property
    def artists(self) -> [Artist]:
        return [a for element in self.all_elements for a in element.artists]

    @property
    def update_artists(self) -> [Artist]:
        return [a for element in self.update_elements for a in element.artists]

    def update(self) -> [Artist]:
        return [a for element in self.update_elements for a in element.update()]

    def remove(self) -> None:
        for artist in self.artists:
            artist.remove()
