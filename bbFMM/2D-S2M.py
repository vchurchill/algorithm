# import necessary packages
import numpy as np
#import matplotlib.pyplot as plt
from functions import *

''' choose source number and node number '''
# N is the number of sources in each interval
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' define boxes '''
# source box
a1 = 0
b1 = 2
a2 = 0
b2 = 2

# target box
c1 = 4
d1 = 6
c2 = 4
d2 = 6

ns1 = np.zeros(shape=(n,1))
nt1 = np.zeros(shape=(n,1))
ns2 = np.zeros(shape=(n,1))
nt2 = np.zeros(shape=(n,1))
for i in range(0,n):
  ns1[i] = nodes(n,a1,b1)[i]
  ns2[i] = nodes(n,a2,b2)[i]
  nt1[i] = nodes(n,c1,d1)[i]
  nt2[i] = nodes(n,c2,d2)[i]

''' populate boxes '''
# target
point = [4.1,4.3]

# sources
sources = np.random.rand(N,2)
for i in range(0,N):
  sources[i,0] = sources[i,0]*(b1-a1)+a1
  sources[i,1] = sources[i,1]*(b2-a2)+a2

''' assign charge to each source, +1 or -1 '''
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

''' weights at Chebyshev nodes in box '''
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,ns1[m1],ns2[m2],sources[j,0],sources[j,1],a1,b1,a2,b2)*sigma[j]
  return sum

''' estimates potential f at Chebyshev nodes '''
def f(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log2(nt1[l1],nt2[l2],ns1[m1],ns2[m2])*W(m1,m2)
  return sum

''' compute potential f at target by interpolation (L2T) '''
fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += f(l1,l2)*R(n,nt1[l1],nt2[l2],point[0],point[1],c1,d1,c2,d2)

print("Estimated potential:")
print(fest)

''' compute actual potential -- sum j=1 to N of f(x)=K(x,y_j)*sigma[j] '''
fact = 0
for j in range(0,N):
  fact += log2(point[0],point[1],sources[j,0],sources[j,1])*sigma[j]

print("Actual potential:")
print(fact)
