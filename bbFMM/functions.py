# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from numpy import *

# Chebyshev polynomial function
def T(n,x,a,b):
  return np.cos(n*np.arccos(((2/(b-a))*(x-((a+b)/2)))))

# compute Chebyshev nodes in interval -1,1
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i] = (a+b)/2+ ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y,a,b):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x,a,b)*T(k,y,a,b)
    k+=1
  return 1/n + (2/n)*sum

# define multivariable version of above S function
def R(n,r1,z1,r2,z2,a,b,c,d):
  return S(n,r1,r2,a,b)*S(n,z1,z2,c,d)

# define log kernel
def log(r,z):
  return np.log(np.fabs(r-z))

# define log kernel for 2D
def log2(r1,z1,r2,z2):
  return np.log(np.sqrt(np.square(r1-r2)+np.square(z1-z2)))
