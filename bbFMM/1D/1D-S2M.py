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

''' source interval '''

a = -1
b = 1

''' target interval '''
c = 11
d = 13

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,1)*(b-a) + a

'''
sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources,sourcesy,color='black')
plt.grid()
plt.show()
'''
# create target
point = 12

# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

# each Chebyshev node, m, gets a weight
def W(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources[j],a,b)*sigma[j]
  return sum

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
def Kest(l):
  Kest = 0
  for m in range(0,n):
    Kest += log(nodes(n,c,d)[l],nodes(n,a,b)[m])*W(m)
  return Kest

fest1 = 0
for l in range(0,n):
  fest1 += S(n,nodes(n,c,d)[l],point,c,d)*Kest(l)

print("Estimated potential 1:")
print(fest1)

# estimate sum to N of f(x)=K(x,y(j))*sigma[j] by sum to n of K(x,y(m))*W(m)
fest2 = 0
for m in range(0,n):
  fest2 += log(point,nodes(n,a,b)[m])*W(m)

print("Estimated potential 2:")
print(fest2)

# compute actual sum to N of f(x)=K(x,y(j))*sigma[j]
fact = 0
for j in range(0,N):
  fact += log(point,sources[j])*sigma[j]

print("Actual potential:")
print(fact)