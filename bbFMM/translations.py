# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import math
import matplotlib.pyplot as plt
from numpy import *

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

# define multivariable version of above S function
def R(n,r1,z1,r2,z2):
  return S(n,r1,r2)*S(n,z1,z2)

# define function to compute chebyshev nodes in interval a,b
def nodes(a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i]=(a+b)/2 + ((b-a)/2)*np.cos(((2*(i+1)-1)*math.pi)/(2*n))
  return nodes

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

plot(0,1,0,1)

#plt.show()

# define multipole-to-multipole and local-to-local translation operators
#

# M2M is a matrix of R_n at specific m1 and m2 in 1,...,n and
# on a particular parent box (a,b)x(c,d)
#
# L2L is a matrix of R_n at specific l1 and l2 in 1,...,n and
# on a particular parent box (a,b)x(c,d)
#
# in each case the child box should be inputed as:
#
# r=1 for bottom left
#   2 for top left
#   3 for bottom right
#   4 for top right
#
# note that valid values for m1, m2, l1, and l2 are 0,...,4 (the possible nodes)
#
# note how there are n^2 M2M matrices per box, one for each
# m1,m2 combo which represents the weight at each of the n^2 nodes in the box


'''
THINK ABOUT for in range(0,n)=# of levels, do M2M(i*a/n,i*b/n,...) for computing all
M2Ms and L2Ls!
'''

def M2M(m1,m2,a,b,c,d,r):
  M = np.zeros(shape=(n,n))
  for i in range(0,n):
    for j in range(0,n):
      # bottom left
      if r==1:
        M[i,j]=R(n,nodes(a,b)[m1],nodes(c,d)[m2],nodes(a,(a+b)/2)[i],nodes(c,(c+d)/2)[j])
      # top left
      if r==2:
        M[i,j]=R(n,nodes(a,b)[m1],nodes(c,d)[m2],nodes(a,(a+b)/2)[i],nodes((c+d)/2,d)[j])
      # bottom right
      if r==3:
        M[i,j]=R(n,nodes(a,b)[m1],nodes(c,d)[m2],nodes((a+b)/2,b)[i],nodes(c,(c+d)/2)[j])
      # top right
      if r==4:
        M[i,j]=R(n,nodes(a,b)[m1],nodes(c,d)[m2],nodes((a+b)/2,b)[i],nodes((c+d)/2,d)[j])
  return M

def L2L(l1,l2,a,b,c,d,r):
  L = np.zeros(shape=(n,n))
  for i in range(0,n):
    for j in range(0,n):
      # parent to bottom left
      if r==1:
        L[i,j]=R(n,nodes(a,(a+b)/2)[l1],nodes(c,(c+d)/2)[l2],nodes(a,b)[i],nodes(c,d)[j])
      # parent to top left
      if r==2:
        L[i,j]=R(n,nodes(a,(a+b)/2)[l1],nodes((c+d)/2,d)[l2],nodes(a,b)[i],nodes(c,d)[j])
      # parent to bottom right
      if r==3:
        L[i,j]=R(n,nodes((a+b)/2,b)[l1],nodes(c,(c+d)/2)[l2],nodes(a,b)[i],nodes(c,d)[j])
      # parent to top right
      if r==4:
        L[i,j]=R(n,nodes((a+b)/2,b)[l1],nodes((c+d)/2,d)[l2],nodes(a,b)[i],nodes(c,d)[j])
  return L

# define kernel-related functions
#
# X is chi
#
# q in 
def X(r1,z1,r2,z2):
  return (np.square(r1)+np.square(r2)+np.square(z1-z2))/(2*r1*r2)

def k(m,r1,z1,r2,z2):
  return (1/np.sqrt(8*np.pi*np.pi*np.pi*r1*r2))*np.sqrt(np.pi)*np.power(2,(-n-1/2))*scipy.special.gamma(n+1/2)*np.power(X(r1,z1,r2,z2),(-n-1/2))*scipy.special.hyp2f1(n/2+3/4,n/2+3/4,n+1,np.power(X(r1,z1,r2,z2),-2))/scipy.special.gamma(n+1)

# define multipole-to-local translation operator
#
# (a,b)x(c,d) is the box we want to calculate the weights for
# (e,f)x(g,h) is the box we already know the weights in
#
# these boxes MUST be well separated!
#
# again there will be 25 weights per box (1 for each node)
#
# m is the number Fourier mode
#

'''
Think about interaction list for M2L, is it only boxes of same size? No!
'''
def M2L(m,k1,k2,a,b,c,d,e,f,g,h):
  K=np.zeros(shape=(n,n))
  if i in range(0,n):
    if j in range(0,n):
      K[i,j]=k(m,nodes(a,b)[k1],nodes(c,d)[k2],nodes(e,f)[i],nodes(g,h)[j])
  return K

# skeletonization stuff
omega1 = (np.pi/n)*np.sqrt(1)


'''
print(L2L(2,3,-1,1,-1,1,1))
print(L2L(2,3,-1,1,-1,1,2))
print(L2L(2,3,-1,1,-1,1,3))
print(L2L(2,3,-1,1,-1,1,4))

print(M2M(2,3,-1,1,-1,1,1))
print(M2M(2,3,-1,1,-1,1,2))
print(M2M(2,3,-1,1,-1,1,3))
print(M2M(2,3,-1,1,-1,1,4))
'''

'''
# define root box
a=-1
b=1
c=-1
d=1
# All M2Ms for root box
for i in range(0,n):
  for j in range(0,n):
    print(M2M(i,j,a,b,c,d))

# All M2Ms for children of root box
for i in range(0,n):
  for j in range(0,n):
    print(M2M(i,j,a,(a+b)/2,c,(c+d)/2))

for i in range(0,n):
  for j in range(0,n):
    print(M2M(i,j,a,(a+b)/2,(c+d)/2,d))

for i in range(0,n):
  for j in range(0,n):
    print(M2M(i,j,(a+b)/2,b,c,(c+d)/2))

for i in range(0,n):
  for j in range(0,n):
    print(M2M(i,j,(a+b)/2,b,(c+d)/2,d))
'''

# other testing
#print(T(5,chebynodes[0]))
#print(T(1,5))
#print(T(1,10))
#print(S(2,5,10))
#print(S(2,3,4))
#print(R(2,3,5,4,10))

