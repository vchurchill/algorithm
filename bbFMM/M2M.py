# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import math
import matplotlib.pyplot as plt
from numpy import *
from functions import *

# of interpolation nodes
n=5

# define Chebyshev polynomial function
def T(n,x):
  if n==0:
    return 1
  if n==1:
    return x
  else:
    return 2*x*T(n-1,x)-T(n-2,x)

# define other Chebyshev polynomial function
def S(n,x,y):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x)*T(k,y)
    k+=1

  return 1/n + (2/n)*sum

# define vector version of S -- Chebyshev polynomial function
def R(n,r1,z1,r2,z2):
  return S(n,r1,r2)*S(n,z1,z2)

# chebyshev nodes in interval a,b
def chebynodes(a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i]=(a+b)/2 + ((b-a)/2)*np.cos(((2*(i+1)-1)*math.pi)/(2*n))
  return nodes

def nodes(a,b):
  nodes =  np.zeros(shape=(n*n,2))
  for j in range(0,n):
    for i in range(0,n):
      nodes[i+j*n,0] = chebynodes(a,b)[i]
      nodes[i+j*n,1] = nodes[j,0]
  return nodes

for i in range(0,n*n):
  plt.scatter(nodes(0,1)[i,0],nodes(0,1)[i,1],color='red')
  plt.scatter(nodes(0,1)[i,0],nodes(1,2)[i,1],color='yellow')
  plt.scatter(nodes(1,2)[i,0],nodes(0,1)[i,1],color='blue')
  plt.scatter(nodes(1,2)[i,0],nodes(1,2)[i,1],color='green')
  plt.scatter(nodes(0,2)[i,0],nodes(0,2)[i,1],color='purple')
plt.grid()
plt.show()

print(nodes(0,1))

def M2M(m1,m2):
  M1 = np.zeros(shape=(n,n))
  M2 = np.zeros(shape=(n,n))
  M3 = np.zeros(shape=(n,n))
  M4 = np.zeros(shape=(n,n))
  for i in range(0,n):
    M1[i,i]=R(n,chebynodes(0,2)[m1],chebynodes(0,2)[m2],nodes(0,1)[i,0],nodes(0,1)[i,1])
    M2[i,i]=R(n,chebynodes(0,2)[m1],chebynodes(0,2)[m2],nodes(0,1)[i,0],nodes(1,2)[i,1])
    M3[i,i]=R(n,chebynodes(0,2)[m1],chebynodes(0,2)[m2],nodes(1,2)[i,0],nodes(0,1)[i,1])
    M4[i,i]=R(n,chebynodes(0,2)[m1],chebynodes(0,2)[m2],nodes(1,2)[i,0],nodes(1,2)[i,1])
  return np.concatenate((M1,M2,M3,M4),axis=0)

print(M2M(3,4))

#print(T(5,chebynodes[0]))
#print(T(1,5))
#print(T(1,10))
#print(S(2,5,10))
#print(S(2,3,4))
#print(R(2,3,5,4,10))