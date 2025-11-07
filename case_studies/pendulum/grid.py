import matplotlib.pyplot as plt
import numpy as np
from pybasin.Sampler import GridSampler


sampler = GridSampler(
    min_limits=(-np.pi + np.arcsin(0.5 / 1.0), -10.0),
    max_limits=(np.pi + np.arcsin(0.5 / 1.0), 10.0),
)

grid = sampler.sample(10000)


plt.figure(figsize=(8, 6))
plt.scatter(grid[:, 0], grid[:, 1], s=1, alpha=0.5)
plt.xlabel("Angle")
plt.ylabel("Force")
plt.title("Grid Sample from GridSampler")
plt.grid(True)
plt.show()
