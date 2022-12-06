from solver import Solver
import numpy as np

def test_calc_f():
    n = 100
    x = 3*np.ones([n, 2])
    u = 2*np.ones([n, 2])
    vd = np.ones([n, 2])
    types = np.ones([n, 1])
    delta_t = 1 
    
    a = Solver(n, x, u, types, delta_t)
    solver_solution = a._Solver__calc_f(vd)
    correct_solution = (vd-u)/a._tau
    assert (abs(correct_solution-solver_solution) < 1e-6).all() == True
    