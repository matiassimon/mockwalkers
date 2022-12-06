# test_solver.py

from solver import Solver


def test_solver_init():
    solver = Solver(
        n=3,
        x=[[2, 6], [7, 8], [-2, 5]],
        u=[[0.5, 0.1], [10, 2], [-1, 4]],
        types=[0, 0, 1],
        delta_t=0.2,
    )
    assert solver.x == [[2, 6], [7, 8], [-2, 5]]
    assert solver.u == [[0.5, 0.1], [10, 2], [-1, 4]]
    assert solver.types == [0, 0, 1]
    assert solver.delta_t == 0.2
