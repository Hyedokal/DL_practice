import numpy as np

Y = np.array([4,5,6])
A = np.array([[2,3,4]]).T
print(Y.shape)
print(A.shape)
print(Y*A)
tmp = -(Y / A) + ((1-Y) / (1-A))
print( tmp.shape )