import numpy as np
from .walkers import Walkers
from .vdcalculator import VdCalculator
from .obstacle import Obstacle


class Solver:
    """
    A class used to represent a Solver

    Attributes
    ----------
    _delta_t : float
        the discrete time step (in seconds) used for the crowd simulation
    _tau : float
        the relaxation time in seconds (default 3)
    _imp: float
        impermeability constant b (default 1)
        impermeability constant rprime (default 0.1)

    Methods
    ----------
    delta_t()
        Returns the float corresponding to the discrete time step (in seconds) used for the crowd simulation
    tau()
        Returns the float corresponding to the relaxation time in seconds (default 3)
    delta_t(delta_t)
        Sets the float corresponding to the discrete time step (in seconds) used for the crowd simulation
    tau(tau)
        Sets the float corresponding to the relaxation time in seconds (default 3)
    """

    def __init__(
        self,
        walkers: Walkers,
        delta_t: float,
        vd_calcs: [VdCalculator],
        obstacles: [Obstacle],
    ):
        """
        Parameters
        ----------
        delta_t : float
            the discrete time step (in seconds) used for the crowd simulation
        """

        self._walkers = walkers
        self._delta_t = delta_t
        self._vd_calcs = vd_calcs
        self._obstacles = obstacles

        # Obstacle interaction constant
        self._obs_constant = float(10)

        # Propulsion constants
        self._tau = float(1)

        # Kernel constants
        self._int_constant = float(1)
        self._int_radius = float(1)
        self._theta_max = np.radians(80)
        self._vel_option = int(1)

        # Iterate constants
        self._current_time = float(0)

        # Propulsion term
        self._f = np.empty(walkers.x.shape)
        self._f.fill(np.nan)

        # Interaction term
        self._ksum = np.empty(walkers.x.shape)
        self._ksum.fill(np.nan)

        # Obstacles term
        self._e = np.empty(walkers.x.shape)
        self._e.fill(np.nan)

    @property
    def walkers(self):
        return self._walkers

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def tau(self):
        return self._tau

    @property
    def int_constant(self):
        return self._int_constant

    @property
    def int_radius(self):
        return self._int_radius

    @property
    def theta_max(self):
        return self._theta_max

    @property
    def vel_option(self):
        return self._vel_option

    @property
    def current_time(self):
        return self._current_time

    @property
    def f(self):
        return self._f

    @property
    def ksum(self):
        return self._ksum

    @property
    def e(self):
        return self._e

    @property
    def vd_calcs(self):
        return self._vd_calcs

    @property
    def obstacles(self):
        return self._obstacles

    @delta_t.setter
    def delta_t(self, delta_t):
        self._delta_t = delta_t
        return self._delta_t

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        return self._tau

    @int_constant.setter
    def int_constant(self, int_constant: float):
        self._int_constant = int_constant
        return self._int_constant

    @int_radius.setter
    def int_radius(self, int_radius: float):
        self._int_radius = int_radius
        return self._int_radius

    @theta_max.setter
    def theta_max(self, theta_max: float):
        self._theta_max = theta_max
        return self._theta_max

    @vel_option.setter
    def vel_option(self, vel_option: int):
        self._vel_option = vel_option
        return self._vel_option

    def __calc_f(self, vd: np.ndarray):
        """
        Calculates propulsion for a desired velocity field vd

        Parameters
        ----------
        vd : ndarray
            the desired velocity field (in meters per second)
        """
        f = (vd - self._walkers.u) / self._tau
        return f

    def __calc_e(self):
        """Implementation of the private method of the Solver class used to calculate
        the E term of the eq. ■, aimed to the corridor example.
        """
        sum = np.zeros(self._walkers.x.shape)

        for obstacle in self._obstacles:
            distance = obstacle.distance(self._walkers)
            distance_mag = np.linalg.norm(distance, axis=1)
            distance_mag = np.broadcast_to(distance_mag[:, np.newaxis], distance.shape)
            sum += (
                distance
                * self._obs_constant
                * np.exp(-distance_mag / obstacle.imp_constant)
                / distance_mag
            )

        return sum

    def __calc_k(self, vd: np.ndarray):
        A = self._int_constant
        R = self._int_radius
        theta_max = self._theta_max

        X = self._walkers.x[:, 0]
        Y = self._walkers.x[:, 1]

        [Xj, Xi] = np.meshgrid(X, X)
        [Yj, Yi] = np.meshgrid(Y, Y)

        distance_vec_bstack = np.stack(
            (np.subtract(Xi, Xj), np.subtract(Yi, Yj)), axis=2
        )
        distance_sqr_nstack = np.square(distance_vec_bstack[:, :, 0]) + np.square(
            distance_vec_bstack[:, :, 1]
        )
        distance_mag_nstack = np.sqrt(distance_sqr_nstack)
        distance_sqr_bstack = np.broadcast_to(
            distance_sqr_nstack[:, :, np.newaxis], (*distance_sqr_nstack.shape, 2)
        )
        distance_mag_bstack = np.broadcast_to(
            distance_mag_nstack[:, :, np.newaxis], (*distance_mag_nstack.shape, 2)
        )

        if self._vel_option == int(0):
            U = vd[:, 0]
            V = vd[:, 1]
        elif self._vel_option == int(1):
            U = self._walkers.u[:, 0]
            V = self._walkers.u[:, 1]

        [Ui, Uj] = np.meshgrid(U, U)
        [Vi, Vj] = np.meshgrid(V, V)

        vd_vec_bstack = np.stack((Uj, Vj), axis=2)
        vd_mag_nstack = np.sqrt(
            np.square(vd_vec_bstack[:, :, 0]) + np.square(vd_vec_bstack[:, :, 1])
        )

        # The distance points TO the i walker, but we need the vector pointing away
        # from it, so we multiply by -1
        product_org = -1 * distance_vec_bstack * vd_vec_bstack
        product_dot = product_org[:, :, 0] + product_org[:, :, 1]
        product_mag = distance_mag_nstack * vd_mag_nstack
        theta_org = np.arccos(product_dot / product_mag)
        theta_con = theta_org < theta_max
        theta = theta_con

        theta_bstack = np.broadcast_to(theta[:, :, np.newaxis], (*theta.shape, 2))

        k = (
            A
            * np.exp((-distance_sqr_bstack) / (R**2))
            * (distance_vec_bstack / distance_mag_bstack)
            * theta_bstack
        )
        k = np.nan_to_num(k)
        return k

    def __calc_vd(self):
        """"""
        return np.sum([vdcalc(self._walkers) for vdcalc in self._vd_calcs], axis=0)

    def iterate(self):
        self._walkers.x = self._walkers.x + self.delta_t * self._walkers.u

        self._vd = self.__calc_vd()
        self._f = self.__calc_f(self._vd)
        self._ksum = np.sum(self.__calc_k(self._vd), axis=1)
        self._e = self.__calc_e()
        self._acceleration = self._f + self._ksum + self._e

        self._walkers.u = self._walkers.u + self._delta_t * self._acceleration

        self._current_time = self._current_time + self._delta_t
