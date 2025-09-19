import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 50)
noise = np.random.normal(0, 2, size=x.shape)  # 均值0, 方差2的噪声
y = 3 * x + 7 + noise
X = np.c_[np.ones_like(x), x]
print(y)
w = np.linalg.inv(X.T @ X) @ X.T @ y
intercept, slope = w
print(f"result: y = {slope:.2f} * x + {intercept:.2f}")
