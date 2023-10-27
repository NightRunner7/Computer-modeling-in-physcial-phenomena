import numpy as np
x = np.array([0.1, 0.2])

y = np.array([[1.1,2.2,3.3],[4.4,5.5,6.6]])


z = y[:, :, np.newaxis] * x

print(z)

print(z[0][0][1])