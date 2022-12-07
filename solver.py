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
    
    def __init__(self, n: int, x: np.ndarray, u: np.ndarray, types, delta_t: float):
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

        self._n = n
        self._x = x
        self._u = u
        self._types = types
        self._delta_t = delta_t
        
        #Propulsion constants
        self._tau = float(3)
        
        #Kernel constants
        self._int_constant = float(1)
        self._int_radius = float(2)
        self._theta_max = np.radians(80)
        
        #Desired velocity constants
        self._vdmag = 1

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
    def types(self, types: np.ndarray):
        self._types = types
        return self._types

    @delta_t.setter
    def delta_t(self, delta_t):
        self._delta_t = delta_t
        return self._delta_t

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        return self._tau

    def __calc_f(self, vd: np.ndarray):
        '''
        Calculates propulsion for a desired velocity field vd

        Parameters
        ----------
        vd : ndarray
            the desired velocity field (in meters per second)
        '''
        f = (vd - self.u)/self.tau
        return f

    
    def __calc_k(self, vd: np.ndarray):
        
        A = self._int_constant
        R = self._int_radius
        theta_max = self._theta_max
        
        X = self._x[:, 0]
        Y = self._x[:, 1]
        
        [Xi, Xj] = np.meshgrid(X, X)
        [Yi, Yj] = np.meshgrid(Y, Y)
        
        distance_vec_bstack = np.stack((np.subtract(Xi, Xj), np.subtract(Yi, Yj)), axis=2)
        distance_sqr_nstack = np.square(distance_vec_bstack[:,:,0]) + np.square(distance_vec_bstack[:,:,1])
        distance_mag_nstack = np.sqrt(distance_sqr_nstack)
        distance_sqr_bstack = np.stack((distance_sqr_nstack, distance_sqr_nstack), axis=2)
        distance_mag_bstack = np.stack((distance_mag_nstack, distance_mag_nstack), axis=2)
        
        U = vd[:, 0]
        V = vd[:, 1]
        
        [Ui, Uj] = np.meshgrid(U, U)
        [Vi, Vj] = np.meshgrid(V, V)
        
        vd_vec_bstack = np.stack((Uj, Vj), axis=2)
        vd_mag_nstack = np.sqrt(np.square(vd_vec_bstack[:,:,0]) + np.square(vd_vec_bstack[:,:,1]))
        
        product_org = distance_vec_bstack * vd_vec_bstack
        product_dot = product_org[:,:,0]+product_org[:,:,1]
        product_mag = distance_mag_nstack * vd_mag_nstack
        theta_org = np.arccos(product_dot/product_mag)
        theta_con = theta_org < theta_max
        theta = theta_org*theta_con
        
        theta_bstack = np.stack((theta, theta), axis=2)
        
        k = A*np.exp((-distance_sqr_bstack)/(R**2))*(distance_vec_bstack/distance_mag_bstack)*theta_bstack
        k = np.nan_to_num(k)
        return k
        
    def __calc_vdterm(self):
        '''Implementation of the private method of the Solver 
        class used to obtain the v_d term for the eq. ■, aimed to the corridor example.
        '''
        vd = np.zeros([self._n, 2])
        vd[:,0] = self._vdmag
        vd[self._types == 1] *= -1 
        return vd
