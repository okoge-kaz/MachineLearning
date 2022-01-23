import numpy as np
import matplotlib.pyplot as plt


Data: np.array = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
X: np.array = Data[:, 0]
Y: np.array = Data[:, 1]
# [:, 0]で[[a1, b1], []...]のa1の方がとれる
# 上記のような表記方法は多用されているのでよく覚えておくこと


class SimpleLinearRegression:
    '''単回帰分析を行うためのクラス'''
    def __init__(self, Data: np.array):
        self.X: np.array = Data[:, 0]
        self.Y: np.array = Data[:, 1]
        # a, bを求める
        self.a: np.float64 = np.cov(X, Y, bias=True)[0][1] / np.var(X, ddof=0)
        # np.covは documentにもあるように、指定されたX, Yからなる共分散行列を返す。そのためCov[X, Y]が欲しい場合は、[0][1] or [1][0]としなくてはならない
        # ddof=0について: 分散を計算する際、平均との偏差の2乗の和をN-ddofで割ります。初期値ではddof=0なのでデータ数であるNで割ることになります。
        self.b: np.float64 = np.average(Y) - self.a * np.average(X)

    def get_a(self) -> np.float64:
        return self.a

    def get_b(self) -> np.float64:
        return self.b

    def get_residual_error(self) -> list[np.float64]:
        response: list[np.float64] = [Y[i] - (a * X[i] + b) for i in range(len(X))]
        return response


simpleLeanearRegression: SimpleLinearRegression = SimpleLinearRegression(Data)
print("a = ", simpleLeanearRegression.get_a())
print("b = ", simpleLeanearRegression.get_b())


a = simpleLeanearRegression.get_a()
b = simpleLeanearRegression.get_b()

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = a * x + b
ax.plot(x, y, label='Simple Regression')
ax.plot(X, Y, '.', label='Data')
plt.legend()
plt.show()


# 残差の計算
residualError: list[np.float64] = simpleLeanearRegression.get_residual_error()
for error in residualError:
    print(error)


print("Cov(X, ε) = ", np.cov(X, residualError, bias=True)[1][0])

print("Cov(Y^, ε) = ", np.cov(a * X + b, residualError, bias=True)[1][0])

print("R^2 = ", 1 - (np.var(residualError, ddof=0)) / (np.var(Y, ddof=0)))
