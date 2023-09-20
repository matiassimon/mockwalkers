from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import mockwalkers as mckw

x0 = np.array([[1, 1], [4, 1]])
u0 = np.array([[0, 0], [0, 0]])
types = np.array([2, 1])
delta_t = 0.005
plot_delta_t = 0.1
final_t = 5

s = None
g = None

fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(-1, 3)


def init_fun():
    global s
    global g
    s = mckw.Solver(2, x0, u0, types, delta_t)
    if g is not None:
        g.remove()
    g = mckw.Graphic(ax, s)


def update(t):
    while s.current_time < t:
        s.iterate()
    g.update()


ani = FuncAnimation(
    fig,
    update,
    init_func=init_fun,
    frames=np.arange(0, final_t, plot_delta_t),
    interval=(1000 * plot_delta_t),
)
plt.show()
