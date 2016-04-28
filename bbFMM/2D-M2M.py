# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''

# N is the number of sources in each interval
N = 10

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' specify source boxes '''
# target box
c1 = 3
d1 = 5
c2 = 3
d2 = 5

# source parent box
a1 = -4
b1 = -2
a2 = -1
b2 = 1

# source children horizontal intervals
C1a1 = a1
C1b1 = (b1+a1)/2
C2a1 = (b1+a1)/2
C2b1 = b1

# source children vertical intervals
C1a2 = a2
C1b2 = (a2+b2)/2
C2a2 = (a2+b2)/2
C2b2 = b2

''' populate boxes '''
# create target
point = [4.1,4.3]

# create sources in child boxes
sources1 = np.random.rand(N,2)
sources2 = np.random.rand(N,2)
sources3 = np.random.rand(N,2)
sources4 = np.random.rand(N,2)
for i in range(0,N):
  sources1[i,0] = sources1[i,0]*(C1b1-C1a1)+C1a1
  sources1[i,1] = sources1[i,1]*(C1b2-C1a2)+C1a2
for i in range(0,N):
  sources2[i,0] = sources2[i,0]*(C1b1-C1a1)+C1a1
  sources2[i,1] = sources2[i,1]*(C2b2-C2a2)+C2a2
for i in range(0,N):
  sources3[i,0] = sources3[i,0]*(C2b1-C2a1)+C2a1
  sources3[i,1] = sources3[i,1]*(C1b2-C1a2)+C1a2
for i in range(0,N):
  sources4[i,0] = sources4[i,0]*(C2b1-C2a1)+C2a1
  sources4[i,1] = sources4[i,1]*(C2b2-C2a2)+C2a2

# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigma1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma1[i] = (np.random.randint(0,2)*2)-1

sigma2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma2[i] = (np.random.randint(0,2)*2)-1

sigma3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma3[i] = (np.random.randint(0,2)*2)-1

sigma4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma4[i] = (np.random.randint(0,2)*2)-1

''' weights for each child boxes '''
def W1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C1a1,C1b1)[m1],nodes(n,C1a2,C1b2)[m2],sources1[j,0],sources1[j,1],C1a1,C1b1,C1a2,C1b2)*sigma1[j]
  return sum

def W2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C1a1,C1b1)[m1],nodes(n,C2a2,C2b2)[m2],sources2[j,0],sources2[j,1],C1a1,C1b1,C2a2,C2b2)*sigma2[j]
  return sum

def W3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C2a1,C2b1)[m1],nodes(n,C1a2,C1b2)[m2],sources3[j,0],sources3[j,1],C2a1,C2b1,C1a2,C1b2)*sigma3[j]
  return sum

def W4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C2a1,C2b1)[m1],nodes(n,C2a2,C2b2)[m2],sources4[j,0],sources4[j,1],C2a1,C2b1,C2a2,C2b2)*sigma4[j]
  return sum

''' weights for parent box '''
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources1[j,0],sources1[j,1],a1,b1,a2,b2)*sigma1[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources2[j,0],sources2[j,1],a1,b1,a2,b2)*sigma2[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources3[j,0],sources3[j,1],a1,b1,a2,b2)*sigma3[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources4[j,0],sources4[j,1],a1,b1,a2,b2)*sigma4[j]
  return sum

''' this is the M2M operation taking the W weights of the child boxes and translating to the parent '''
def West(m1,m2):
  sum = 0
  for mprime1 in range(0,n):
    for mprime2 in range(0,n):
      sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C1a1,C1b1)[mprime1],nodes(n,C1a2,C1b2)[mprime2],a1,b1,a2,b2)*W1(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C1a1,C1b1)[mprime1],nodes(n,C2a2,C2b2)[mprime2],a1,b1,a2,b2)*W2(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C2a1,C2b1)[mprime1],nodes(n,C1a2,C1b2)[mprime2],a1,b1,a2,b2)*W3(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C2a1,C2b1)[mprime1],nodes(n,C2a2,C2b2)[mprime2],a1,b1,a2,b2)*W4(mprime1,mprime2)
  return sum

print("Difference between estimate and actual weights for each (m1,m2) node:")
for m1 in range(0,n):
  for m2 in range(0,n):
    print(W(m1,m2)-West(m1,m2))