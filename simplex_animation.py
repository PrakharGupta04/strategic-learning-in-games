import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load history
history = np.load("fp_history.npy")

# Convert to simplex coordinates
def to_simplex(p):
    x = p[1] + 0.5 * p[2]
    y = (np.sqrt(3)/2) * p[2]
    return x, y

coords = np.array([to_simplex(p) for p in history])

fig, ax = plt.subplots()

# Triangle
triangle = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2],
    [0, 0]
])
ax.plot(triangle[:,0], triangle[:,1], 'k-')

point, = ax.plot([], [], 'ro', label="Current")
trail, = ax.plot([], [], 'b-', alpha=0.6, label="Path")

ax.set_title("Fictitious Play Trajectory on Simplex")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1)
ax.legend()

def update(frame):
    x, y = coords[frame]
    point.set_data([x], [y])   # ✅ FIXED BUG HERE
    trail.set_data(coords[:frame,0], coords[:frame,1])
    return point, trail

ani = FuncAnimation(fig, update, frames=len(coords), interval=30)

plt.show()