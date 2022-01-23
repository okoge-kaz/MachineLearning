import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


# 論理積 AND

X: np.array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y: np.array = np.array([0, 0, 0, 1])
mAND: SGDClassifier = SGDClassifier(loss='log')
mAND.fit(X, Y)
# SGDClassifier loss = 'log' : 'log' means 'logistic regression'

print(mAND.predict(X))


def draw_decision_area(ax, model, X, vmin, vmax):
    w1, w2 = model.coef_[0]
    bias = model.intercept_[0]
    a = -w1 / w2
    b = -bias / w2
    if bias <= 0:
        ax.fill_between(X, vmin, a * X + b, color='tab:blue', hatch='.', alpha=0.3)
        ax.fill_between(X, a * X + b, vmax, color='tab:red', hatch='o', alpha=0.3)
    else:
        ax.fill_between(X, vmin, a * X + b, color='tab:red', hatch='o', alpha=0.3)
        ax.fill_between(X, a * X + b, vmax, color='tab:blue', hatch='.', alpha=0.3)


def draw_points(ax, Xp, Yp):
    colors = np.array(["tab:blue", "tab:red"])
    I = np.where(Yp == 0)
    ax.scatter(Xp[I, 0], Xp[I, 1], c='black', marker='_')
    I = np.where(Yp == 1)
    ax.scatter(Xp[I, 0], Xp[I, 1], c='black', marker='+')


def show_decisions(model, Xp, Yp, N=400):
    vmin = -0.05
    vmax = 1.05
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    X = np.linspace(vmin, vmax, N)
    draw_decision_area(ax, model, X, vmin, vmax)
    draw_points(ax, Xp, Yp)
    plt.show()


show_decisions(mAND, X, Y)

print(mAND.coef_, end=", ")
print(mAND.intercept_)

# 論理和 OR

X: np.array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y: np.array = np.array([0, 1, 1, 1])
mOR = SGDClassifier(loss='log')
mOR.fit(X, Y)

print(mOR.predict(X))

show_decisions(mOR, X, Y)

print(mOR.coef_, mOR.intercept_)

