# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
from translations import *
from functions import *
import matplotlib.pyplot as plt

# N is the number of sources in each box
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

# 2*rb is the side length of each box
# for now, the box sizes are the same
rb = 0.5

# x,y is the position of the bottom left hand corner of the box
# change these to get different boxes
x1 = 0
y1 = 0

# create sources in box
sources = np.random.rand(N,2) + [x1, y1]

# create point outside box 1
point = [5.2,5.2]

# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(1,N))
for i in range(0,N):
  sigma[0,i] = (np.random.randint(0,2)*2)-1
print(sigma)
# initialize source coordinate vectors
rs = np.zeros(shape=(N,1))
zs = np.zeros(shape=(N,1))

# break down source points into source coordinate vectors
for i in range(0,N):
  rs[i] = sources[i,0]
  zs[i] = sources[i,1]

# check that this is actually points in a box with a circle around it
# sources
plt.scatter(rs,zs,color='black')
# point
plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()

def P1(m1,m2,a,b,c,d):
  P1 = np.zeros(shape=(N,1))
  for i in range(0,n):
    P1[i] = R(n,nodes(n,a,b)[m1],nodes(n,c,d)[m2],rs[i],zs[i])
  return P1

W = np.zeros(shape=(1,n*n))
for i in range(0,n):
  for j in range(0,n):
    W[0,j+5*i] = np.dot(sigma,P1(i,j,0,1,0,1))

def kernel(l1,l2):
  K = np.zeros(shape=(n*n,1))
  for i in range(0,n):
    for j in range(0,n):
      K[j+5*i] = log(nodes(n,5,5)[l1],nodes(n,5,5)[l2],nodes(n,0,1)[i],nodes(n,0,1)[j])
  return K

F = np.zeros(shape=(1,n*n))
for i in range(0,n):
  for j in range(0,n):
    F[0,j+5*i]=np.dot(W,kernel(i,j))

P2 = np.zeros(shape=(n*n,1))
for i in range(0,n):
  for j in range(0,n):
    P2[j+5*i] = R(n,nodes(n,5,5)[i],nodes(n,5,5)[j],point[0],point[1])

f = np.dot(F,P2)

'''
The above was the estimate through bbFMM process
Now we fine the actual potential
'''

Kactual = np.zeros(shape=(1,N))
for i in range(0,N):
  Kactual[0,i] = log(point[0],point[1],rs[i],zs[i])

factual = np.dot(Kactual,sigma.T)

print("Actual:")
print(factual)
print("Estimate:")
print(f)
