# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import scipy
from scipy import special
import math
import matplotlib.pyplot as plt

# 2*rb is the side length of each box
# for now, the box sizes are the same
rb = 0.125

# x,y is the position of the bottom left hand corner of the box
# change these to get different boxes
x1 = 0
y1 = 0
x2 = 0.5
y2 = 0

# c is the center of each box
c1 = [rb + x1, rb + y1]
c2 = [rb + x2, rb + y2]

# p is the number of discretization points on the surfaces
p = 7

# box 1 check surface
rc1 = np.zeros(shape=(p,1))
zc1 = np.zeros(shape=(p,1))
# box 1 equiv surface
rq1 = np.zeros(shape=(p,1))
zq1 = np.zeros(shape=(p,1))
# box 2 equiv surface
rq2 = np.zeros(shape=(p,1))
zq2 = np.zeros(shape=(p,1))

K1 = np.zeros(shape=(p,p))
K2 = np.zeros(shape=(p,p))

# create discretization points on surfaces
# define surface constant
d=0.1
# radii for surfaces
radius1 = (4-math.sqrt(2)-2*d)*rb
radius2 = (math.sqrt(2)+d)*rb

# box 1 upward check surface and downward equivalent surface points
for i in range(0,p):
  rc1[i] = radius1 * math.cos(math.pi*2 * i/p) + c1[0]
  zc1[i] = radius1 * math.sin(math.pi*2 * i/p) + c1[1]

# box 1 upward equivalent surface and downward check surface points
for i in range(0,p):
  rq1[i] = radius2 * math.cos(math.pi*2 * i/p) + c1[0]
  zq1[i] = radius2 * math.sin(math.pi*2 * i/p) + c1[1]

# box 2 upward equivalent surface and downward check surface points
for i in range(0,p):
  rq2[i] = radius2 * math.cos(math.pi*2 * i/p) + c2[0]
  zq2[i] = radius2 * math.sin(math.pi*2 * i/p) + c2[1]

# define Green's function
def G(r1,z1,r2,z2):
  return (1/(2*math.pi))*np.log(math.sqrt(np.square(r1-r2)+np.square(z1-z2)))

# calculate kernel matrices
# green's function at rq1,zq1 on down check surf & rq2,zq2 on up equiv surface
# p x p matrix, used to find down check potential for box 1
for i in range(0,p):
  for j in range(0,p):
    K1[i,j] = G(rq1[i],zq1[i],rq2[j],zq2[j])

# green's function at rq1,zq1 on down check surf & rc1,zc1 on down equiv surf
# p x p matrix, used to solve integral eqn for down equiv density of box 1
for i in range(0,p):
  for j in range(0,p):
    K2[i,j] = G(rq1[i],zq1[i],rc1[j],zc1[j])

# define tikhonov regularization function
# regularization parameter
alpha = np.power(10,-12)
# identity matrix
I = np.matlib.identity(p)
def tikh(M):
  return np.dot(np.linalg.inv(alpha*I+np.dot(np.matrix.transpose(M),M)),np.matrix.transpose(M))

# The M2L operator is this matrix:
# (p/(2*math.pi*radius1))*[(alpha*I + K2^T*K2)^(-1)]*K2^T*K1

M2L = (p/(2*math.pi*radius1))*np.dot(tikh(K2),K1)

#import sys
#sys.stdout = open("/Users/HomeBase/Github/algorithm/output.txt","w")
print(M2L)