import mockwalkers as mckw
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Obstracles
obstacles = [mckw.RectangleObstacle(-1, 0, 1, 1), mckw.RectangleObstacle(5, 0, 1, 1)]

# Walkers positions
x = np.array([[1, 0.5], [4, 0.5]])
# Walkers types
type_1_flag = 0x01
type_2_flag = 0x02
types = np.array([type_1_flag, type_2_flag])

# Desired velocities
type_1_vd = (1, 0)
type_2_vd = (-1, 0)
vd_calcs = [
    mckw.RectangleVelocityBooster(0, 0, 5, 1, *type_1_vd, type_1_flag),
    mckw.RectangleVelocityBooster(0, 0, 5, 1, *type_2_vd, type_2_flag),
]

# Simulation paramters
delta_t = 0.01
final_t = 15

# Plot configuration
repeat = True
plot_delta_t = 0.1

s = None
g = None

fig, ax = plt.subplots()
ax.set_xlim(-1, 6)
ax.set_ylim(0, 1)
ax.set_aspect("equal")


def create_solver():
    return mckw.Solver(
        mckw.Walkers(x, np.zeros((2, 2)), types),
        delta_t,
        vd_calcs,
        obstacles,
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
# ani.save("corridor_example.gif")
