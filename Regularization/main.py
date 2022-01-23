import numpy as np
import matplotlib.pyplot as plt


X = np.array([0., 0.16, 0.22, 0.34, 0.44, 0.5, 0.67, 0.73, 0.9, 1.])
Y = np.array([-0.06, 0.94, 0.97, 0.85, 0.25, 0.09, -0.9, -0.93, -0.53, 0.08])

fig, ax = plt.subplots(dpi=100)
ax.scatter(X, Y, marker='o', color='b')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.grid()
plt.show()

W3 = np.polyfit(X, Y, 3)
print(W3)

x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(dpi=100)
ax.scatter(X, Y, marker='o', color='b')
ax.plot(x, np.polyval(W3, x), 'r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.grid()
plt.show()

print("平均二乗残差: ", end='')
print(np.mean((Y - np.polyval(W3, X)) ** 2))

W9 = np.polyfit(X, Y, 9)
print(W9)

x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(dpi=100)
ax.scatter(X, Y, marker='o', color='b')
ax.plot(x, np.polyval(W9, x), 'r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_ylim(-1.2, 1.2)
ax.grid()
plt.show()

x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(dpi=100)
ax.scatter(X, Y, marker='o', color='b')
ax.plot(x, np.polyval(W9, x), 'r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.grid()
plt.show()

print("平均二乗残差: ", end='')
print(np.mean((Y - np.polyval(W9, X)) ** 2))

