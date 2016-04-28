# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from functions import *

''' specify number of sources per box and number of nodes per interval '''

# N is the number of sources in each box
N = 4

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=10

''' specify boxes '''

# target box
Pa1 = 2
Pb1 = 4
Pa2 = 2
Pb2 = 4

# interaction list boxes
P1c1 = 0
P1d1 = 2
P1c2 = 6
P1d2 = 8

P2c1 = 2
P2d1 = 4
P2c2 = 6
P2d2 = 8

P3c1 = 4
P3d1 = 6
P3c2 = 6
P3d2 = 8

P4c1 = 6
P4d1 = 8
P4c2 = 6
P4d2 = 8

P5c1 = 6
P5d1 = 8
P5c2 = 4
P5d2 = 6

P6c1 = 6
P6d1 = 8
P6c2 = 2
P6d2 = 4

P7c1 = 6
P7d1 = 8
P7c2 = 0
P7d2 = 2

''' populate boxes '''
# create target
point = [3,3]

# create sources
sourcesP1 = np.random.rand(N,2)
sourcesP2 = np.random.rand(N,2)
sourcesP3 = np.random.rand(N,2)
sourcesP4 = np.random.rand(N,2)
sourcesP5 = np.random.rand(N,2)
sourcesP6 = np.random.rand(N,2)
sourcesP7 = np.random.rand(N,2)

# put sources in our interaction list boxes
for i in range(0,N):
  sourcesP1[i,0] = sourcesP1[i,0]*(P1d1-P1c1)+P1c1
  sourcesP1[i,1] = sourcesP1[i,1]*(P1d2-P1c2)+P1c2
for i in range(0,N):
  sourcesP2[i,0] = sourcesP2[i,0]*(P2d1-P2c1)+P2c1
  sourcesP2[i,1] = sourcesP2[i,1]*(P2d2-P2c2)+P2c2
for i in range(0,N):
  sourcesP3[i,0] = sourcesP3[i,0]*(P3d1-P3c1)+P3c1
  sourcesP3[i,1] = sourcesP3[i,1]*(P3d2-P3c2)+P3c2
for i in range(0,N):
  sourcesP4[i,0] = sourcesP4[i,0]*(P4d1-P4c1)+P4c1
  sourcesP4[i,1] = sourcesP4[i,1]*(P4d2-P4c2)+P4c2
for i in range(0,N):
  sourcesP5[i,0] = sourcesP5[i,0]*(P5d1-P5c1)+P5c1
  sourcesP5[i,1] = sourcesP5[i,1]*(P5d2-P5c2)+P5c2
for i in range(0,N):
  sourcesP6[i,0] = sourcesP6[i,0]*(P6d1-P6c1)+P6c1
  sourcesP6[i,1] = sourcesP6[i,1]*(P6d2-P6c2)+P6c2
for i in range(0,N):
  sourcesP7[i,0] = sourcesP7[i,0]*(P7d1-P7c1)+P7c1
  sourcesP7[i,1] = sourcesP7[i,1]*(P7d2-P7c2)+P7c2

'''
# plot sources
for i in range(0,N):
  plt.scatter(sourcesP1[i,0],sourcesP1[i,1],color='black')
  plt.scatter(sourcesP2[i,0],sourcesP2[i,1],color='red')
  plt.scatter(sourcesP3[i,0],sourcesP3[i,1],color='blue')
  plt.scatter(sourcesP4[i,0],sourcesP4[i,1],color='yellow')
  plt.scatter(sourcesP5[i,0],sourcesP5[i,1],color='black')
  plt.scatter(sourcesP6[i,0],sourcesP6[i,1],color='red')
  plt.scatter(sourcesP7[i,0],sourcesP7[i,1],color='blue')
  plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()
'''

''' assign charge (+ or - 1) to sources in each int. list box'''
sigmaP1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP1[i] = (np.random.randint(0,2)*2)-1

sigmaP2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP2[i] = (np.random.randint(0,2)*2)-1
  
sigmaP3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP3[i] = (np.random.randint(0,2)*2)-1

sigmaP4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP4[i] = (np.random.randint(0,2)*2)-1

sigmaP5 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP5[i] = (np.random.randint(0,2)*2)-1

sigmaP6 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP6[i] = (np.random.randint(0,2)*2)-1

sigmaP7 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaP7[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each interaction list interaction '''

# each Chebyshev node, m, gets a weight
def WP1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P1c1,P1d1)[m1],nodes(n,P1c2,P1d2)[m2],sourcesP1[j,0],sourcesP1[j,1],P1c1,P1d1,P1c2,P1d2)*sigmaP1[j]
  return sum

def WP2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P2c1,P2d1)[m1],nodes(n,P2c2,P2d2)[m2],sourcesP2[j,0],sourcesP2[j,1],P2c1,P2d1,P2c2,P2d2)*sigmaP2[j]
  return sum

def WP3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P3c1,P3d1)[m1],nodes(n,P3c2,P3d2)[m2],sourcesP3[j,0],sourcesP3[j,1],P3c1,P3d1,P3c2,P3d2)*sigmaP3[j]
  return sum

def WP4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P4c1,P4d1)[m1],nodes(n,P4c2,P4d2)[m2],sourcesP4[j,0],sourcesP4[j,1],P4c1,P4d1,P4c2,P4d2)*sigmaP4[j]
  return sum

def WP5(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P5c1,P5d1)[m1],nodes(n,P5c2,P5d2)[m2],sourcesP5[j,0],sourcesP5[j,1],P5c1,P5d1,P5c2,P5d2)*sigmaP5[j]
  return sum

def WP6(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P6c1,P6d1)[m1],nodes(n,P6c2,P6d2)[m2],sourcesP6[j,0],sourcesP6[j,1],P6c1,P6d1,P6c2,P6d2)*sigmaP6[j]
  return sum

def WP7(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,P7c1,P7d1)[m1],nodes(n,P7c2,P7d2)[m2],sourcesP7[j,0],sourcesP7[j,1],P7c1,P7d1,P7c2,P7d2)*sigmaP7[j]
  return sum

''' compute M2L contribution. in this case this is also local exp. since 
there is no contribution from L2L since the int. list of the parent box
doesn't exist'''
def g(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P1c1,P1d1)[m1],nodes(n,P1c2,P1d2)[m2])*WP1(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P2c1,P2d1)[m1],nodes(n,P2c2,P2d2)[m2])*WP2(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P3c1,P3d1)[m1],nodes(n,P3c2,P3d2)[m2])*WP3(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P4c1,P4d1)[m1],nodes(n,P4c2,P4d2)[m2])*WP4(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P5c1,P5d1)[m1],nodes(n,P5c2,P5d2)[m2])*WP5(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P6c1,P6d1)[m1],nodes(n,P6c2,P6d2)[m2])*WP6(m1,m2) + log2(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P7c1,P7d1)[m1],nodes(n,P7c2,P7d2)[m2])*WP7(m1,m2)
  return sum

''' L2P contribution '''
fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += g(l1,l2)*R(n,nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],point[0],point[1],Pa1,Pb1,Pa2,Pb2)

print("Estimated potential:")
print(fest)

''' computation of actual local expansion '''
fact = 0
for j in range(0,N):
  fact += log2(sourcesP1[j,0],sourcesP1[j,1],point[0],point[1])*sigmaP1[j] + log2(sourcesP2[j,0],sourcesP2[j,1],point[0],point[1])*sigmaP2[j] + log2(sourcesP3[j,0],sourcesP3[j,1],point[0],point[1])*sigmaP3[j] + log2(sourcesP4[j,0],sourcesP4[j,1],point[0],point[1])*sigmaP4[j] + log2(sourcesP5[j,0],sourcesP5[j,1],point[0],point[1])*sigmaP5[j] + log2(sourcesP6[j,0],sourcesP6[j,1],point[0],point[1])*sigmaP6[j] + log2(sourcesP7[j,0],sourcesP7[j,1],point[0],point[1])*sigmaP7[j]

print("Actual potential:")
print(fact)