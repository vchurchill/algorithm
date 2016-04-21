# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from numpy import *

# Chebyshev polynomial function
def T(n,x):
  if n==0:
    return 1
  if n==1:
    return x
  else:
    return 2*x*T(n-1,x)-T(n-2,x)

# computes chebyshev nodes in interval a,b
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i]=(a+b)/2 + ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes


# define other Chebyshev polynomial function
def S(n,x,y):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x)*T(k,y)
    k+=1
  return 1/n + (2/n)*sum

# define multivariable version of above S function
def R(n,r1,z1,r2,z2):
  return S(n,r1,r2)*S(n,z1,z2)

# define log kernel
def log(r1,z1,r2,z2):
  return np.log(np.sqrt(np.square(r1-r2)+np.square(z1-z2)))


'''
# plot the interpolation nodes in 2D
#
# a,b is the x interval and c,d is the y interval
def plot(a,b,c,d):
  for i in range(0,n):
    for j in range(0,n):
      plt.scatter(nodes(a,(a+b)/2)[i],nodes(c,(c+d)/2)[j],color='red')
      plt.scatter(nodes(a,(a+b)/2)[i],nodes((c+d)/2,d)[j],color='yellow')
      plt.scatter(nodes((a+b)/2,b)[i],nodes(c,(c+d)/2)[j],color='blue')
      plt.scatter(nodes((a+b)/2,b)[i],nodes((c+d)/2,d)[j],color='green')
      plt.scatter(nodes(a,b)[i],nodes(c,d)[j],color='purple')
  plt.grid()
  plt.show()

plot(0,1,0,1)
'''