import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.animation
import japanize_matplotlib


D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])

colormap = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
offset_y = [0.5, 0.5, -0.5, -0.5]
xmin, xmax = 0, 10
ymin, ymax = 0, 10


def init_graph(dpi=100, grid=True):
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 6))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    if grid:
        ax.grid()
    return fig, ax
# dpi = dots per inch : figsizeはグラフサイズを表す。
# ax.set_aspect('equal') : Set the aspect ratio of the axes scaling, i.e. y/x-scale.
# reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html


def plot_data(ax, D, offset_y=None, marker='o'):
    A = []
    for i, row in enumerate(D):
        A.append(ax.scatter(D[i, 0], D[i, 1], marker=marker, color=colormap[i]))
        if offset_y is not None:
            A.append(ax.text(D[i, 0], D[i, 1] + offset_y[i], f'$(x_{i+1}, y_{i+1})$', ha='center'))
    return A
# matplotli.markers : https://matplotlib.org/stable/api/markers_api.html
# marker=にどのような値を渡すと、どのようなプロットになるのかが書かれている


fig, ax = init_graph()
plot_data(ax, D, offset_y)
plt.show()

offset_hat_y = [-0.8, -0.8, 0.6, 0.6]


def plot_line(ax, a, b, label=None, color='black'):
    if label is None:
        label = f'${a}x + {b}$' if a != 1 else f'$x + {b}$'
    x = np.array([xmin, xmax])
    return ax.plot(x, a * x + b, color, ls='-', label=label)


def plot_hat_y(ax, D, a, b, offset_hat_y, marker='*', show_text=True):
    A = []
    for i, row in enumerate(D):
        x = D[i, 0]
        hat_y = a * x + b
        A.append(ax.scatter(x, hat_y, marker=marker, color=colormap[i]))
        if show_text:
            A.append(ax.text(x, hat_y + offset_hat_y[i], f'$(x_{i+1}, \\hat{{y}}_{i+1})$', ha='center'))
    return A


fig, ax = init_graph()
plot_data(ax, D, offset_y)
plot_line(ax, 1, 1)
plot_hat_y(ax, D, 1, 1, offset_hat_y)
plt.legend(loc='upper left')
plt.show()
