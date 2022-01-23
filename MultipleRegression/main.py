import numpy as np
import matplotlib.pyplot as plt


a, b = 0.431, 3.310
D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])

fig, ax = plt.subplots(dpi=100, figsize=(6, 6))

ax.scatter(D[:, 0], D[:, 1], marker='o')

x = np.linspace(0, 10, 500)
ax.plot(x, 0.09 * x ** 3 - 1.27 * x ** 2 + 5.41 * x - 1.23, 'tab:red', ls='-', label='$0.09 x^3 - 1.27 x^2 + 5.41 x - 1.23$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.grid()
plt.legend(loc='upper left')
plt.show()

