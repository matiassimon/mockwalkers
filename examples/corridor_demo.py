import mockwalkers as mckw
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

n_walkers = 100
corridor_width = 40
corridor_height = 7
x = np.random.rand(n_walkers, 2)
x[:, 0] *= corridor_width
x[:, 1] *= corridor_height
types = np.concatenate(
    (np.array([0x01]).repeat(n_walkers // 2), np.array([0x02]).repeat(n_walkers // 2))
)
walkers = mckw.Walkers(x, np.zeros((n_walkers, 2)), types)
delta_t = 0.001
vd_calcs = [
    mckw.RectangleVelocityBooster(-corridor_width, 0, 3*corridor_width, corridor_height, 1, 0, 0x01),
    mckw.RectangleVelocityBooster(-corridor_width, 0, 3*corridor_width, corridor_height, -1, 0, 0x02),
]
obstacles = [
    mckw.RectangleObstacle(0, 0, corridor_width, -1),
    mckw.RectangleObstacle(0, corridor_height, corridor_width, 1),
]
plot_delta_t = 0.1
final_t = 30
repeat = False

s = None
g = None

fig, ax = plt.subplots()
ax.set_xlim(-1, corridor_width + 1)
ax.set_ylim(-1, corridor_height + 1)

def init_fun():
    global s
    global g
    s = mckw.Solver(walkers, delta_t, vd_calcs, obstacles)
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
