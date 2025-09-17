import numpy as np
import pandas as pd

# 读取数据
salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')

# 取前30行
y = salary['Salary'][:30].values
x = salary[['Experience Years']][:30].values

# 构造设计矩阵 X (加一列全1)
X = np.c_[np.ones((30,1)), x]

# 正规方程求解 w = (X^T X)^(-1) X^T y
w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# 解包参数
inter, w1 = w
print(f"回归方程: y = {w1:.2f} * x + {inter:.2f}")
