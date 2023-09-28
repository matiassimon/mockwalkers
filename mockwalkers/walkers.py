from typing import Union
import numpy as np


class Walkers:
    """
    Attributes
    ----------
    _n : int
        the number of individuals
    _x : ndarray
        an n x 2 array containing the two-dimensional positions of the individuals
    _u : ndarray
        an n x 2 array containing the two-dimensional velocities of the individuals
    _types : ndarray
        an n x 1 array containing the types of individuals

    Methods
    ----------
    x()
        Returns the array _x containing the two-dimensional positions of the individuals
    u()
        Returns the array _u containing the two-dimensional velocities of the individuals
    types()
        Returns the array _types containing the types of individuals
    x(x)
        Sets the array _x containing the two-dimensional positions of the individuals
    u(u)
        Sets the array _u containing the two-dimensional velocities of the individuals
    types(types)
        Sets the array _types containing the types of individuals in the crowd
    """

    def __init__(self, x: np.ndarray, u: np.ndarray, types: np.ndarray = None) -> None:
        """
        Parameters
        ----------
        x : ndarray
            an n x 2 array containing the two-dimensional positions of the individuals
        u : ndarray
            an n x 2 array containing the two-dimensional velocities of the individuals
        types : ndarray
            an n x 1 array containing the types of individuals in the crowd
        """
        # Check if the shape of x is (N, 2) where N != 0
        if x.shape != (x.shape[0], 2) or x.shape[0] == 0:
            raise ValueError("x should have shape (N, 2) where N != 0")

        # Check if the shapes of x and u are different
        if x.shape != u.shape:
            raise ValueError("the shapes of x and u must be the same")

        # Check if the shape of types is (N,)
        if types is not None and types.shape != (x.shape[0],):
            raise ValueError("types should have shape (N,)")

        self._n = x.shape[0]
        self._x = x
        self._u = u
        self._types = types
        self._int_radius = float(1)
        self._theta_max = np.radians(80)

    @property
    def n(self) -> int:
        return self._n

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def u(self) -> np.ndarray:
        return self._u

    @property
    def types(self) -> Union(None, np.ndarray):
        return self._types

    @property
    def int_radius(self) -> float:
        return self._int_radius

    @property
    def theta_max(self) -> float:
        return self._theta_max

    @x.setter
    def x(self, x: np.ndarray) -> np.ndarray:
        if x.shape != self._x.shape:
            raise ValueError("the shape of x cannot be changed")
        self._x = x
        return self._x

    @u.setter
    def u(self, u) -> np.ndarray:
        if u.shape != self._u.shape:
            raise ValueError("the shape of u cannot be changed")
        self._u = u
        return self._u

    @types.setter
    def types(self, types: np.ndarray) -> np.ndarray:
        if types.shape != self._types.shape:
            raise ValueError("the shape of types cannot be changed")
        self._types = types
        return self._types

    @int_radius.setter
    def int_radius(self, int_radius: float) -> float:
        self._int_radius = int_radius
        return self._int_radius

    @theta_max.setter
    def theta_max(self, theta_max: float) -> float:
        self._theta_max = theta_max
        return self._theta_max
