import numpy as np

class Solver:
    '''
    A class used to represent a Solver
    
    ...
    
    Attributes
    ----------
    _n : int
        the number of individuals for the crowd simulation
    _x : ndarray
        an n x 2 array containing the two-dimensional positions of the individuals
    _u : ndarray
        an n x 2 array containing the two-dimensional velocities of the individuals
    _types : ndarray
        an n x 1 array containing the types of individuals in the crowd
    _delta_t : float
        the discrete time step (in seconds) used for the crowd simulation
    _tau : float
        the relaxation time in seconds (default 3)
        
    Methods
    ----------
    x()
        Returns the array _x containing the two-dimensional positions of the individuals
    u()
        Returns the array _u containing the two-dimensional velocities of the individuals
    types()
        Returns the array _types containing the types of individuals in the crowd
    delta_t()
        Returns the float corresponding to the discrete time step (in seconds) used for the crowd simulation
    tau()
        Returns the float corresponding to the relaxation time in seconds (default 3)
    x(x)
        Sets the array _x containing the two-dimensional positions of the individuals
    u(u)
        Sets the array _u containing the two-dimensional velocities of the individuals
    types(types)
        Sets the array _types containing the types of individuals in the crowd
    delta_t(delta_t)
        Sets the float corresponding to the discrete time step (in seconds) used for the crowd simulation
    tau(tau)
        Sets the float corresponding to the relaxation time in seconds (default 3)
    '''
    
    def __init__(self, n: int, x, u, types, delta_t: float):
        '''
        Parameters
        -------pyth---
        n : int
            the number of individuals for the crowd simulation
        x : ndarray
            an n x 2 array containing the two-dimensional positions of the individuals
        u : ndarray
            an n x 2 array containing the two-dimensional velocities of the individuals
        types : ndarray
            an n x 1 array containing the types of individuals in the crowd
        delta_t : float
            the discrete time step (in seconds) used for the crowd simulation
        '''
        tau = float(3)
        
        self._n = n
        self._x = x
        self._u = u
        self._types = types
        self._delta_t = delta_t
        self._vdmag = 1
        self._tau = tau

    @property
    def x(self):
        return self._x

    @property
    def u(self):
        return self._u

    @property
    def types(self):
        return self._types

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def tau(self):
       return self._tau

    @x.setter
    def x(self, x):
        self._x = x
        return self._x

    @u.setter
    def u(self, u):
        self._u = u
        return self._u

    @types.setter
    def types(self, types):
        self._types = types
        return self._types

    @delta_t.setter
    def delta_t(self, delta_t):
        self._delta_t = delta_t
        return self._delta_t

    def __calc_vdterm(self):
        '''Implementation of the private method of the Solver 
        class used to obtain the v_d term for the eq. ■, aimed to the corridor example.
        '''
        vd = np.zeroes([self._n, 2])
        vd[:1] = self._vdmag
        vd[self._types == 1] *= -1 
        return vd
    

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        return self._tau

    def __calc_f(self, vd):
        '''
        Calculates propulsion for a desired velocity field vd

        Parameters
        ----------
        vd : ndarray
            the desired velocity field (in meters per second)
        '''
        f = (vd - self.u)/self.tau
        return f
