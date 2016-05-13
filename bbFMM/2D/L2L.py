# import necessary packages
import numpy as np
#import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''
# N is the number of sources in each interval
N = 3

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' define boxes '''
# target box
a1 = 3
b1 = 4
a2 = 3
b2 = 4

# parent box
Pa1 = 2
Pb1 = 4
Pa2 = 2
Pb2 = 4

# parent interaction list boxes
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

nt1 = np.zeros(shape=(n,1))
nt2 = np.zeros(shape=(n,1))
np1 = np.zeros(shape=(n,1))
np2 = np.zeros(shape=(n,1))
nc11 = np.zeros(shape=(n,1))
nc12 = np.zeros(shape=(n,1))
nc21 = np.zeros(shape=(n,1))
nc22 = np.zeros(shape=(n,1))
nc31 = np.zeros(shape=(n,1))
nc32 = np.zeros(shape=(n,1))
nc41 = np.zeros(shape=(n,1))
nc42 = np.zeros(shape=(n,1))
nc51 = np.zeros(shape=(n,1))
nc52 = np.zeros(shape=(n,1))
nc61 = np.zeros(shape=(n,1))
nc62 = np.zeros(shape=(n,1))
nc71 = np.zeros(shape=(n,1))
nc72 = np.zeros(shape=(n,1))

for i in range(0,n):
  nt1[i] = nodes(n,a1,b1)[i]
  nt2[i] = nodes(n,a2,b2)[i]
  np1[i] = nodes(n,Pa1,Pb1)[i]
  np2[i] = nodes(n,Pa2,Pb2)[i]
  nc11[i] = nodes(n,P1c1,P1d1)[i]
  nc12[i] = nodes(n,P1c2,P1d2)[i]
  nc21[i] = nodes(n,P2c1,P2d1)[i]
  nc22[i] = nodes(n,P2c2,P2d2)[i]
  nc31[i] = nodes(n,P3c1,P3d1)[i]
  nc32[i] = nodes(n,P3c2,P3d2)[i]
  nc41[i] = nodes(n,P4c1,P4d1)[i]
  nc42[i] = nodes(n,P4c2,P4d2)[i]
  nc51[i] = nodes(n,P5c1,P5d1)[i]
  nc52[i] = nodes(n,P5c2,P5d2)[i]
  nc61[i] = nodes(n,P6c1,P6d1)[i]
  nc62[i] = nodes(n,P6c2,P6d2)[i]
  nc71[i] = nodes(n,P7c1,P7d1)[i]
  nc72[i] = nodes(n,P7c2,P7d2)[i]

''' populate boxes '''
# create target
point = [3.1,3.3]

# create sources
sourcesP1 = np.random.rand(N,2)
sourcesP2 = np.random.rand(N,2)
sourcesP3 = np.random.rand(N,2)
sourcesP4 = np.random.rand(N,2)
sourcesP5 = np.random.rand(N,2)
sourcesP6 = np.random.rand(N,2)
sourcesP7 = np.random.rand(N,2)

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

''' assign charge (+ or - 1) to sources in each int. list box for parent'''
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
    sum += R(n,nc11[m1],nc12[m2],sourcesP1[j,0],sourcesP1[j,1],P1c1,P1d1,P1c2,P1d2)*sigmaP1[j]
  return sum

def WP2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc21[m1],nc22[m2],sourcesP2[j,0],sourcesP2[j,1],P2c1,P2d1,P2c2,P2d2)*sigmaP2[j]
  return sum

def WP3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc31[m1],nc32[m2],sourcesP3[j,0],sourcesP3[j,1],P3c1,P3d1,P3c2,P3d2)*sigmaP3[j]
  return sum

def WP4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc41[m1],nc42[m2],sourcesP4[j,0],sourcesP4[j,1],P4c1,P4d1,P4c2,P4d2)*sigmaP4[j]
  return sum

def WP5(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc51[m1],nc52[m2],sourcesP5[j,0],sourcesP5[j,1],P5c1,P5d1,P5c2,P5d2)*sigmaP5[j]
  return sum

def WP6(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc61[m1],nc62[m2],sourcesP6[j,0],sourcesP6[j,1],P6c1,P6d1,P6c2,P6d2)*sigmaP6[j]
  return sum

def WP7(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc71[m1],nc72[m2],sourcesP7[j,0],sourcesP7[j,1],P7c1,P7d1,P7c2,P7d2)*sigmaP7[j]
  return sum

''' compute multipole to local estimate for parent '''
def fP(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log2(np1[l1],np2[l2],nc11[m1],nc12[m2])*WP1(m1,m2) + log2(np1[l1],np2[l2],nc21[m1],nc22[m2])*WP2(m1,m2) + log2(np1[l1],np2[l2],nc31[m1],nc32[m2])*WP3(m1,m2) + log2(np1[l1],np2[l2],nc41[m1],nc42[m2])*WP4(m1,m2) + log2(np1[l1],np2[l2],nc51[m1],nc52[m2])*WP5(m1,m2) + log2(np1[l1],np2[l2],nc61[m1],nc62[m2])*WP6(m1,m2) + log2(np1[l1],np2[l2],nc71[m1],nc72[m2])*WP7(m1,m2)
  return sum

''' compute estimate for local expansion note the M2L target box contribution is 0 '''
def fT(l1,l2):
  sum = 0
  for lprime1 in range(0,n):
    for lprime2 in range(0,n):
      sum += fP(lprime1,lprime2)*R(n,nt1[l1],nt2[l2],np1[lprime1],np2[lprime2],Pa1,Pb1,Pa2,Pb2)
  return sum

''' L2T operation '''
fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += fT(l1,l2)*R(n,nt1[l1],nt2[l2],point[0],point[1],a1,b1,a2,b2)

print("Estimated potential:")
print(fest)

''' computation of actual local expansion '''
fact = 0
for j in range(0,N):
  fact += log2(point[0],point[1],sourcesP1[j,0],sourcesP1[j,1])*sigmaP1[j] + log2(point[0],point[1],sourcesP2[j,0],sourcesP2[j,1])*sigmaP2[j] + log2(point[0],point[1],sourcesP3[j,0],sourcesP3[j,1])*sigmaP3[j] + log2(point[0],point[1],sourcesP4[j,0],sourcesP4[j,1])*sigmaP4[j] + log2(point[0],point[1],sourcesP5[j,0],sourcesP5[j,1])*sigmaP5[j] + log2(point[0],point[1],sourcesP6[j,0],sourcesP6[j,1])*sigmaP6[j] + log2(point[0],point[1],sourcesP7[j,0],sourcesP7[j,1])*sigmaP7[j]

print("Actual potential:")
print(fact)