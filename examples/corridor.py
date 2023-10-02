import mockwalkers as mckw
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

np.random.seed(42)

# Corridor geometry
corridor_width = 30
corridor_height = 6
obstacles = [
    mckw.RectangleObstacle(0, 0, corridor_width, -1),
    mckw.RectangleObstacle(0, corridor_height, corridor_width, 1),
]
# Periodic geometry
geometry = mckw.EuclideanXPeriodic(0, corridor_width)

# Number of walkers
n_walkers = 70
# Walkers random initial positions
x = np.random.rand(n_walkers, 2)
x[:, 0] *= corridor_width
x[:, 1] *= corridor_height
# Walkers types
type_1_flag = 0x01
type_2_flag = 0x02
# Assign types to each walker
types = np.concatenate(
    (
        np.array([type_1_flag]).repeat(n_walkers // 2 + n_walkers % 2),
        np.array([type_2_flag]).repeat(n_walkers // 2),
    )
)
walkers = mckw.Walkers(x, np.zeros((n_walkers, 2)), types)

# Desired velocities
type_1_vd = (1, 0)
type_2_vd = (-1, 0)
vd_calcs = [
    mckw.RectangleVelocityBooster(
        0, 0, corridor_width, corridor_height, *type_1_vd, type_1_flag
    ),
    mckw.RectangleVelocityBooster(
        0, 0, corridor_width, corridor_height, *type_2_vd, type_2_flag
    ),
]

# Simulation paramters
delta_t = 0.01
final_t = 50

# Plot configuration
repeat = False
plot_delta_t = 0.1

s = None
g = None

fig, ax = plt.subplots(figsize=(12,4))
ax.set_xlim(-1, corridor_width + 1)
ax.set_ylim(-1, corridor_height + 1)
ax.set_aspect("equal")


def create_solver():
    return mckw.Solver(
        walkers,
        delta_t,
        vd_calcs,
        obstacles,
        geometry,
    )


def create_graphic():
    return mckw.Graphic(ax, s, f_arrows=True, ksum_arrows=True, e_arrows=True)


def init_fun():
    global s
    global g
    s = create_solver()
    if g is not None:
        g.remove()
    g = create_graphic()
    return g.update_artists


def update(t):
    while s.current_time < t:
        s.iterate()
    return g.update()


ani = FuncAnimation(
    fig,
    update,
    init_func=init_fun,
    frames=np.arange(0, final_t, plot_delta_t),
    interval=(1000 * plot_delta_t),
    repeat=repeat,
    blit=True,
)

plt.show()
# ani.save("corridor.gif")
