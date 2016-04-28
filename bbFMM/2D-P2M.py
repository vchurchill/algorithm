# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''

# N is the number of sources in each interval
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' specify source interval '''

a1 = 0
b1 = 2
a2 = -1
b2 = 1
c1 = 4
d1 = 6
c2 = 4
d2 = 6

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,2)
for i in range(0,N):
  sources[i,0] = sources[i,0]*(b1-a1)+a1
  sources[i,1] = sources[i,1]*(b2-a2)+a2

# create target
point = [4.1,4.3]

for i in range(0,N):
  plt.scatter(sources[i,0],sources[i,1],color='black')
  plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()

# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each box '''

# each Chebyshev node, m, gets a weight
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources[j,0],sources[j,1],a1,b1,a2,b2)*sigma[j]
  return sum

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
def Kest(l1,l2):
  Kest = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      Kest += log2(nodes(n,c1,d1)[l1],nodes(n,c2,d2)[l2],nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2])*W(m1,m2)
  return Kest

''' L2T contribution '''
fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += R(n,nodes(n,c1,d1)[l1],nodes(n,c2,d2)[l2],point[0],point[1],c1,d1,c2,d2)*Kest(l1,l2)

print("Estimated potential:")
print(fest)

# compute actual sum to N of f(x)=K(x,y(j))*sigma[j]
fact = 0
for j in range(0,N):
  fact += log2(point[0],point[1],sources[j,0],sources[j,1])*sigma[j]

print("Actual potential:")
print(fact)
