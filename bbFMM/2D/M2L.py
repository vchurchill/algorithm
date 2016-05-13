# import necessary packages
import numpy as np
#import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''
# N is the number of sources in each box
N = 3

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' define boxes '''
# target box
a1 = 2
b1 = 4
a2 = 2
b2 = 4

# interaction list boxes
I1c1 = 0
I1d1 = 2
I1c2 = 6
I1d2 = 8

I2c1 = 2
I2d1 = 4
I2c2 = 6
I2d2 = 8

I3c1 = 4
I3d1 = 6
I3c2 = 6
I3d2 = 8

I4c1 = 6
I4d1 = 8
I4c2 = 6
I4d2 = 8

I5c1 = 6
I5d1 = 8
I5c2 = 4
I5d2 = 6

I6c1 = 6
I6d1 = 8
I6c2 = 2
I6d2 = 4

I7c1 = 6
I7d1 = 8
I7c2 = 0
I7d2 = 2

nt1 = np.zeros(shape=(n,1))
nt2 = np.zeros(shape=(n,1))
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
  nc11[i] = nodes(n,I1c1,I1d1)[i]
  nc12[i] = nodes(n,I1c2,I1d2)[i]
  nc21[i] = nodes(n,I2c1,I2d1)[i]
  nc22[i] = nodes(n,I2c2,I2d2)[i]
  nc31[i] = nodes(n,I3c1,I3d1)[i]
  nc32[i] = nodes(n,I3c2,I3d2)[i]
  nc41[i] = nodes(n,I4c1,I4d1)[i]
  nc42[i] = nodes(n,I4c2,I4d2)[i]
  nc51[i] = nodes(n,I5c1,I5d1)[i]
  nc52[i] = nodes(n,I5c2,I5d2)[i]
  nc61[i] = nodes(n,I6c1,I6d1)[i]
  nc62[i] = nodes(n,I6c2,I6d2)[i]
  nc71[i] = nodes(n,I7c1,I7d1)[i]
  nc72[i] = nodes(n,I7c2,I7d2)[i]

''' populate boxes '''
# create target
point = [3.7,3.8]

# create sources
sourcesI1 = np.random.rand(N,2)
sourcesI2 = np.random.rand(N,2)
sourcesI3 = np.random.rand(N,2)
sourcesI4 = np.random.rand(N,2)
sourcesI5 = np.random.rand(N,2)
sourcesI6 = np.random.rand(N,2)
sourcesI7 = np.random.rand(N,2)
for i in range(0,N):
  sourcesI1[i,0] = sourcesI1[i,0]*(I1d1-I1c1)+I1c1
  sourcesI1[i,1] = sourcesI1[i,1]*(I1d2-I1c2)+I1c2
for i in range(0,N):
  sourcesI2[i,0] = sourcesI2[i,0]*(I2d1-I2c1)+I2c1
  sourcesI2[i,1] = sourcesI2[i,1]*(I2d2-I2c2)+I2c2
for i in range(0,N):
  sourcesI3[i,0] = sourcesI3[i,0]*(I3d1-I3c1)+I3c1
  sourcesI3[i,1] = sourcesI3[i,1]*(I3d2-I3c2)+I3c2
for i in range(0,N):
  sourcesI4[i,0] = sourcesI4[i,0]*(I4d1-I4c1)+I4c1
  sourcesI4[i,1] = sourcesI4[i,1]*(I4d2-I4c2)+I4c2
for i in range(0,N):
  sourcesI5[i,0] = sourcesI5[i,0]*(I5d1-I5c1)+I5c1
  sourcesI5[i,1] = sourcesI5[i,1]*(I5d2-I5c2)+I5c2
for i in range(0,N):
  sourcesI6[i,0] = sourcesI6[i,0]*(I6d1-I6c1)+I6c1
  sourcesI6[i,1] = sourcesI6[i,1]*(I6d2-I6c2)+I6c2
for i in range(0,N):
  sourcesI7[i,0] = sourcesI7[i,0]*(I7d1-I7c1)+I7c1
  sourcesI7[i,1] = sourcesI7[i,1]*(I7d2-I7c2)+I7c2

''' assign charge (+ or - 1) to sources in each int. list box'''
sigmaI1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI1[i] = (np.random.randint(0,2)*2)-1

sigmaI2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI2[i] = (np.random.randint(0,2)*2)-1
  
sigmaI3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI3[i] = (np.random.randint(0,2)*2)-1

sigmaI4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI4[i] = (np.random.randint(0,2)*2)-1

sigmaI5 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI5[i] = (np.random.randint(0,2)*2)-1

sigmaI6 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI6[i] = (np.random.randint(0,2)*2)-1

sigmaI7 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaI7[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each interaction list interaction '''
def WI1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc11[m1],nc12[m2],sourcesI1[j,0],sourcesI1[j,1],I1c1,I1d1,I1c2,I1d2)*sigmaI1[j]
  return sum

def WI2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc21[m1],nc22[m2],sourcesI2[j,0],sourcesI2[j,1],I2c1,I2d1,I2c2,I2d2)*sigmaI2[j]
  return sum

def WI3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc31[m1],nc32[m2],sourcesI3[j,0],sourcesI3[j,1],I3c1,I3d1,I3c2,I3d2)*sigmaI3[j]
  return sum

def WI4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc41[m1],nc42[m2],sourcesI4[j,0],sourcesI4[j,1],I4c1,I4d1,I4c2,I4d2)*sigmaI4[j]
  return sum

def WI5(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc51[m1],nc52[m2],sourcesI5[j,0],sourcesI5[j,1],I5c1,I5d1,I5c2,I5d2)*sigmaI5[j]
  return sum

def WI6(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc61[m1],nc62[m2],sourcesI6[j,0],sourcesI6[j,1],I6c1,I6d1,I6c2,I6d2)*sigmaI6[j]
  return sum

def WI7(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc71[m1],nc72[m2],sourcesI7[j,0],sourcesI7[j,1],I7c1,I7d1,I7c2,I7d2)*sigmaI7[j]
  return sum

''' M2L contribution. in this case this is the whole local exp. since 
there is no contribution from L2L since there is no int. list of the parent box'''
def g(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log2(nt1[l1],nt2[l2],nc11[m1],nc12[m2])*WI1(m1,m2) + log2(nt1[l1],nt2[l2],nc21[m1],nc22[m2])*WI2(m1,m2) + log2(nt1[l1],nt2[l2],nc31[m1],nc32[m2])*WI3(m1,m2) + log2(nt1[l1],nt2[l2],nc41[m1],nc42[m2])*WI4(m1,m2) + log2(nt1[l1],nt2[l2],nc51[m1],nc52[m2])*WI5(m1,m2) + log2(nt1[l1],nt2[l2],nc61[m1],nc62[m2])*WI6(m1,m2) + log2(nt1[l1],nt2[l2],nc71[m1],nc72[m2])*WI7(m1,m2)
  return sum

''' L2T computation '''
fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += g(l1,l2)*R(n,nt1[l1],nt2[l2],point[0],point[1],a1,b1,a2,b2)

print("Estimated potential:")
print(fest)

''' computation of actual local expansion '''
fact = 0
for j in range(0,N):
  fact += log2(sourcesI1[j,0],sourcesI1[j,1],point[0],point[1])*sigmaI1[j] + log2(sourcesI2[j,0],sourcesI2[j,1],point[0],point[1])*sigmaI2[j] + log2(sourcesI3[j,0],sourcesI3[j,1],point[0],point[1])*sigmaI3[j] + log2(sourcesI4[j,0],sourcesI4[j,1],point[0],point[1])*sigmaI4[j] + log2(sourcesI5[j,0],sourcesI5[j,1],point[0],point[1])*sigmaI5[j] + log2(sourcesI6[j,0],sourcesI6[j,1],point[0],point[1])*sigmaI6[j] + log2(sourcesI7[j,0],sourcesI7[j,1],point[0],point[1])*sigmaI7[j]

print("Actual potential:")
print(fact)