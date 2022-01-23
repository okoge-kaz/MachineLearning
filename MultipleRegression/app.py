import numpy as np
import matplotlib.pyplot as plt


Data: np.array = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])


class Regression():

    def __init__(self, data, degree):
        self.X = data[:, 0].reshape(1, -1)
        # [1, n]の行列に変換
        self.X_design = np.array([np.power(self.X, i) for i in range(degree + 1)]).reshape(degree + 1, -1).T
        # degree 次関数による重回帰を行いたい
        self.Y = data[:, 1].reshape(-1, 1)
        # [n, 1]の行列に変換
        self.W = np.dot(np.dot(np.linalg.pinv(np.dot(self.X_design.T, self.X_design)), self.X_design.T), self.Y).reshape(-1)
        # (2.25)式を利用する
        self.Y_prediction = np.polyval(self.W[::-1], self.X)
        # numpy.polyval の仕様上の都合から 重みWを反転させておく必要がある

    def get_W(self):
        return self.W

    def get_R2(self):
        # 式 (2.49) numpy varianceを用いる
        return np.var(self.Y_prediction, ddof=0) / np.var(self.Y, ddof=0)

    def plot_graph(self):
        x = np.linspace(-20, 30, 500)
        y = np.polyval(self.W[::-1], x)  # 重みベクトルWとベクトルxからyを求める(推測値)
        fig, ax = plt.subplots(dpi=500)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-20, 30)
        ax.set_ylim(0, 13000)
        ax.grid()
        ax.scatter(self.X, self.Y, marker=".")
        ax.plot(x, y, "g")
        fig.show()

    def plot_data(self):
        fig, ax = plt.subplots(dpi=500)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-20, 30)
        ax.set_ylim(0, 13000)
        ax.grid()
        ax.scatter(self.X, self.Y, marker=".")
        fig.show()
