# import necessary packages
import numpy as np
#import matplotlib.pyplot as plt
from functions import *

''' specify source number and node number '''
# N is the number of sources in each interval
N = 10

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' define source boxes '''
# source parent box
a1 = 0
b1 = 2
a2 = 0
b2 = 2

# source horizontal intervals
C1a1 = a1
C1b1 = (b1+a1)/2
C2a1 = (b1+a1)/2
C2b1 = b1

# source vertical intervals
C1a2 = a2
C1b2 = (a2+b2)/2
C2a2 = (a2+b2)/2
C2b2 = b2

ns1 = np.zeros(shape=(n,1))
ns2 = np.zeros(shape=(n,1))
nc1 = np.zeros(shape=(n,1))
nc2 = np.zeros(shape=(n,1))
nc3 = np.zeros(shape=(n,1))
nc4 = np.zeros(shape=(n,1))
for i in range(0,n):
  ns1[i]=nodes(n,a1,b1)[i]
  ns2[i]=nodes(n,a2,b2)[i]
  nc1[i]=nodes(n,C1a1,C1b1)[i]
  nc2[i]=nodes(n,C2a1,C2b1)[i]
  nc3[i]=nodes(n,C1a2,C1b2)[i]
  nc4[i]=nodes(n,C2a2,C2b2)[i]

''' populate boxes '''
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

''' assign charge (+ or - 1) to sources in each box'''
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

''' weights for nodes of each child boxes '''
def W1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc1[m1],nc3[m2],sources1[j,0],sources1[j,1],C1a1,C1b1,C1a2,C1b2)*sigma1[j]
  return sum

def W2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc1[m1],nc4[m2],sources2[j,0],sources2[j,1],C1a1,C1b1,C2a2,C2b2)*sigma2[j]
  return sum

def W3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc2[m1],nc3[m2],sources3[j,0],sources3[j,1],C2a1,C2b1,C1a2,C1b2)*sigma3[j]
  return sum

def W4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nc2[m1],nc4[m2],sources4[j,0],sources4[j,1],C2a1,C2b1,C2a2,C2b2)*sigma4[j]
  return sum

''' weights for nodes of parent box '''
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,ns1[m1],ns2[m2],sources1[j,0],sources1[j,1],a1,b1,a2,b2)*sigma1[j] + R(n,ns1[m1],ns2[m2],sources2[j,0],sources2[j,1],a1,b1,a2,b2)*sigma2[j] + R(n,ns1[m1],ns2[m2],sources3[j,0],sources3[j,1],a1,b1,a2,b2)*sigma3[j] + R(n,ns1[m1],ns2[m2],sources4[j,0],sources4[j,1],a1,b1,a2,b2)*sigma4[j]
  return sum

''' M2M operation turning weights of the child boxes into weight of parent '''
def WP(m1,m2):
  sum = 0
  for mprime1 in range(0,n):
    for mprime2 in range(0,n):
      sum += R(n,ns1[m1],ns2[m2],nc1[mprime1],nc3[mprime2],a1,b1,a2,b2)*W1(mprime1,mprime2) + R(n,ns1[m1],ns2[m2],nc1[mprime1],nc4[mprime2],a1,b1,a2,b2)*W2(mprime1,mprime2) + R(n,ns1[m1],ns2[m2],nc2[mprime1],nc3[mprime2],a1,b1,a2,b2)*W3(mprime1,mprime2) + R(n,ns1[m1],ns2[m2],nc2[mprime1],nc4[mprime2],a1,b1,a2,b2)*W4(mprime1,mprime2)
  return sum

print("Difference between estimate and actual weights for each (m1,m2) node:")
for m1 in range(0,n):
  for m2 in range(0,n):
    print((m1,m2))
    print(W(m1,m2)-WP(m1,m2))