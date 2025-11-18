import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 设置中文字体（Windows 通用）
plt.rcParams['font.family'] = 'SimHei'       # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 构造数据
np.random.seed(0)
X = np.linspace(-3, 3, 30)
y = X**3 - X + np.random.randn(30) * 3
X = X.reshape(-1, 1)

# 定义多项式模型函数
def fit_poly_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return y_pred

# 拟合并可视化
for d in [1, 3, 15]:  # 欠拟合、正常拟合、过拟合
    y_pred = fit_poly_model(d)
    plt.plot(X, y_pred, label=f'阶数 {d}（{"欠拟合" if d==1 else "正常拟合" if d==3 else "过拟合"}）')

plt.scatter(X, y, color='black', label='原始数据')
plt.legend()
plt.title("不同模型复杂度下的拟合表现")
plt.xlabel("输入特征 X")
plt.ylabel("目标值 y")
plt.grid(True)
plt.show()
