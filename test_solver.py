from solver import Solver
from pytest import approx
import numpy as np

def test_calc_f():
    ''''
    Unit test for the function calculating the propulsion F
    '''
    
    # Set parameters to some chosen values
    n = 100
    x = 3*np.ones([n, 2])
    u = 2*np.ones([n, 2])
    vd = np.ones([n, 2])
    types = np.ones([n, 1])
    delta_t = 1 
    
    a = Solver(n, x, u, types, delta_t)
    solver_solution = a._Solver__calc_f(vd)
    correct_solution = (vd-u)/a._tau
    
    # Check if the solver solution and the correct_solution are approximately equal
    assert approx(solver_solution) == correct_solution
    