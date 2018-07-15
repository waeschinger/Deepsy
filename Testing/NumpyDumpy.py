import numpy as np
import matplotlib as plt


x = np.random.rand(3, 4)
print(x)

y = np.ones([3, 4], dtype=float)

print(y)

y.T.dot(x)

print(y)
