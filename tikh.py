import numpy as np
from numpy import matlib
from numpy import linalg

# define tikhonov regularization function
def tikh(M,p):
  # regularization parameter
  alpha = np.power(10,-12)
  # identity matrix
  I = np.matlib.identity(p)
  return np.dot(np.linalg.inv(alpha*I+np.dot(np.matrix.transpose(M),M)),np.matrix.transpose(M))
