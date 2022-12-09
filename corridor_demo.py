# This file is part of MockWalkers.
# 
# MockWalkers is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# 
# MockWalkers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with MockWalkers.
# If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import streamlit.components.v1 as components
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import solver
import graphic
import time

n = 50
u0 = np.zeros((n,2))
#types = np.ones((n,1))
types = np.ones((n,1))
types[n//2:] = 2

def gen_rand_x():
    x = np.random.rand(n,2)
    x[:,0] *= solver.CORRIDOR_LENGTH
    x[:,1] *= solver.CORRIDOR_WIDTH
    return x

s = solver.Solver(n, gen_rand_x(), u0, types, 0.01)
fig, ax = plt.subplots()
g = graphic.Graphic(ax, s)
g.traces_on = True


st_plot = st.pyplot(fig)
st_time_label = st.text("Time: -")

def update(i):
    s.iterate()
    if i%5 == 0:
        g.update(s)
        st_plot.pyplot(fig)
        st_time_label.write(f"Time: {s.current_time:.3f} s")
        time.sleep(s.delta_t)

for i in range(1000):
    update(i)
    
