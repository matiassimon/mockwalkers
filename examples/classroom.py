import mockwalkers as mckw
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

rng = np.random.default_rng()

wall_thick = 0.3
classroom_width = 6.6
classroom_height = 6.6
door_edge = 0.3
door_width = 1.2
single_desk_width = 1.2
double_desk_width = 2.1
desk_height = 0.6
desk_hspace = (classroom_width - 2 * single_desk_width - double_desk_width) / 2
desk_vspace = 0.6
vdmag = 1

walls = [
    mckw.RectangleObstacle(
        -wall_thick, -wall_thick, classroom_width + 2 * wall_thick, wall_thick
    ),
    mckw.RectangleObstacle(
        -wall_thick, classroom_height, classroom_width + 2 * wall_thick, wall_thick
    ),
    mckw.RectangleObstacle(
        -wall_thick, 0, wall_thick, classroom_height - door_width - door_edge
    ),
    mckw.RectangleObstacle(
        -wall_thick, classroom_height - door_edge, wall_thick, door_edge
    ),
    mckw.RectangleObstacle(classroom_width, 0, wall_thick, classroom_height),
]


def create_desk_row(y):
    return (
        mckw.RectangleObstacle(0, y, single_desk_width, desk_height),
        mckw.RectangleObstacle(
            single_desk_width + desk_hspace, y, double_desk_width, desk_height
        ),
        mckw.RectangleObstacle(
            single_desk_width + 2 * desk_hspace + double_desk_width,
            y,
            single_desk_width,
            desk_height,
        ),
    )


desks = [
    *create_desk_row(desk_vspace),
    *create_desk_row(2 * desk_vspace + desk_height),
    *create_desk_row(3 * desk_vspace + 2 * desk_height),
    *create_desk_row(4 * desk_vspace + 3 * desk_height),
]


def create_boost_row(y):
    return (
        mckw.RectangleVelocityBooster(0, y, single_desk_width, desk_vspace, vdmag, 0),
        mckw.RectangleVelocityBooster(
            single_desk_width + desk_hspace,
            y,
            double_desk_width,
            desk_vspace,
            -vdmag,
            0,
        ),
        mckw.RectangleVelocityBooster(
            single_desk_width + 2 * desk_hspace + double_desk_width,
            y,
            single_desk_width,
            desk_vspace,
            -vdmag,
            0,
        ),
    )


boosters = [
    *create_boost_row(0),
    *create_boost_row(1 * desk_vspace + desk_height),
    *create_boost_row(2 * desk_vspace + 2 * desk_height),
    *create_boost_row(3 * desk_vspace + 3 * desk_height),
    mckw.RectangleVelocityBooster(
        single_desk_width,
        0,
        desk_hspace,
        desk_height * 4 + desk_vspace * 4,
        0,
        vdmag,
    ),
    mckw.RectangleVelocityBooster(
        single_desk_width + desk_hspace + double_desk_width,
        0,
        desk_hspace,
        desk_height * 4 + desk_vspace * 4,
        0,
        vdmag,
    ),
    mckw.RectangleVelocityBooster(
        0,
        classroom_height - 2 * door_edge - door_width,
        classroom_width,
        door_edge,
        0,
        vdmag,
    ),
    mckw.RectangleVelocityBooster(
        0,
        classroom_height - door_edge,
        classroom_width,
        door_edge,
        0,
        -vdmag,
    ),
    mckw.RectangleVelocityBooster(
        -6,
        classroom_height - door_edge - door_width,
        classroom_width + 6,
        door_width,
        -vdmag,
        0,
    ),
]


def create_walkers_row(y):
    return np.array(
        [
            [single_desk_width * 1 / 2, y],
            [single_desk_width + desk_hspace + double_desk_width * 2 / 7, y],
            [single_desk_width + desk_hspace + double_desk_width * 5 / 7, y],
            [
                single_desk_width
                + 2 * desk_hspace
                + double_desk_width
                + single_desk_width * 1 / 2,
                y,
            ],
        ]
    )


walkers_x = np.concatenate(
    [
        create_walkers_row(desk_vspace * 1 / 2),
        create_walkers_row(desk_height + desk_vspace * 3 / 2),
        create_walkers_row(2 * desk_height + desk_vspace * 5 / 2),
        create_walkers_row(3 * desk_height + desk_vspace * 7 / 2),
    ]
)
walkers_speeds = rng.normal(1, 0.2, 16)

walkers = mckw.Walkers(walkers_x, np.zeros((16, 2)), None, walkers_speeds)

# Simulation paramters
delta_t = 0.01
final_t = 20

# Plot configuration
repeat = False
plot_delta_t = 0.1

s = None
g = None

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, classroom_width + 1)
ax.set_ylim(-1, classroom_height + 1)
ax.set_aspect("equal")


def create_solver():
    return mckw.Solver(
        walkers,
        delta_t,
        boosters,
        [*walls, *desks],
    )


def create_graphic():
    return mckw.Graphic(
        ax,
        s,
        f_arrows=True,
        ksum_arrows=True,
        e_arrows=True,
        traces_kwargs={"size": 100, "skip": 2},
    )


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
# ani.save("classroom.gif")
