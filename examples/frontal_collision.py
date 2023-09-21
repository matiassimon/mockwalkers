import mockwalkers as mckw
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

walkers = mckw.Walkers(
    np.array([[1, 1], [4, 1]]), np.array([[0, 0], [0, 0]]), np.array([0x01, 0x02])
)
delta_t = 0.005
vd_calcs = [
    mckw.RectangleVelocityBooster(0, 0, 10, 10, 1, 0, 0x01),
    mckw.RectangleVelocityBooster(0, 0, 10, 10, -1, 0, 0x02),
]
plot_delta_t = 0.1
final_t = 5
repeat = False

s = None
g = None

fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(-1, 3)


def init_fun():
    global s
    global g
    s = mckw.Solver(walkers, delta_t, vd_calcs)
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
    repeat=repeat,
)
plt.show()
