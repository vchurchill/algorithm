# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
from functions import *
import math
import matplotlib.pyplot as plt

# N is the number of sources in each box
N1 = 20
N2 = 20

# p is the number of discretization points on the surfaces
p = 16

# 2*rb is the side length of each box
# for now, the box sizes are the same
rb = 0.5

# x,y is the position of the bottom left hand corner of the box
# change these to get different boxes
x1 = 0
y1 = 0
x2 = 2
y2 = 0

# c is the center of each box
c1 = [rb + x1, rb + y1]
c2 = [rb + x2, rb + y2]

# create sources for box 1
sources1 = np.random.rand(N1,2)
sources1 = sources1 + [x1, y1]

# create sources for box 2
sources2 = np.random.rand(N2,2)
sources2 = sources2 + [x2, y2]

# create some density for each source. Ying says this is "given"
# note: I wasn't sure what to put as these values
# just did random integers from 0 to 10
phi1 = np.random.randint(10, size=(N1,1))
phi2 = np.random.randint(10, size=(N2,1))

# initialize coordinate vectors
# box 1 sources
rs1 = np.zeros(shape=(N1,1))
zs1 = np.zeros(shape=(N1,1))
# box 1 check surface
rc1 = np.zeros(shape=(p,1))
zc1 = np.zeros(shape=(p,1))
# box 1 equiv surface
rq1 = np.zeros(shape=(p,1))
zq1 = np.zeros(shape=(p,1))
# box 2 sources
rs2 = np.zeros(shape=(N2,1))
zs2 = np.zeros(shape=(N2,1))
# box 2 check surface
rc2 = np.zeros(shape=(p,1))
zc2 = np.zeros(shape=(p,1))
# box 3 equiv surface
rq2 = np.zeros(shape=(p,1))
zq2 = np.zeros(shape=(p,1))

# initiate kernel matrices
K1 = np.zeros(shape=(p,N1))
K2 = np.zeros(shape=(p,N2))
K3 = np.zeros(shape=(p,p))
K4 = np.zeros(shape=(p,p))
K5 = np.zeros(shape=(p,p))
K6 = np.zeros(shape=(p,p))

# initialize upward check potential vectors
q1u = np.zeros(shape=(p,1))
q2u = np.zeros(shape=(p,1))

# initialize downward check potential vectors
q1d = np.zeros(shape=(p,1))

# break down source points into source coordinate vectors
for i in range(0,N1):
  rs1[i] = sources1[i,0]
  zs1[i] = sources1[i,1]
  
for i in range(0,N2):
  rs2[i] = sources2[i,0]
  zs2[i] = sources2[i,1]

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

# box 2 upward check surface and downward equivalent surface points
for i in range(0,p):
  rc2[i] = radius1 * math.cos(math.pi*2 * i/p) + c2[0]
  zc2[i] = radius1 * math.sin(math.pi*2 * i/p) + c2[1]

# box 2 upward equivalent surface and downward check surface points
for i in range(0,p):
  rq2[i] = radius2 * math.cos(math.pi*2 * i/p) + c2[0]
  zq2[i] = radius2 * math.sin(math.pi*2 * i/p) + c2[1]

# check that this is actually points in a box with a circle around it
# sources
plt.scatter(rs1,zs1,color='black')
plt.scatter(rs2,zs2,color='black')
# upward equivalent surfaces
plt.scatter(rq1,zq1,color='red')
plt.scatter(rq2,zq2,color='red')
# upward check surfaces
plt.scatter(rc1,zc1,color='blue')
plt.scatter(rc2,zc2,color='blue')
plt.show()
# uncomment the line above to show the plot
# to continue the program after the plot, you must close the plot window

# calculate kernel matrices

# green's function at rt1,zt1 on the upward check surface and rs1,zs1 sources
# p x N1 matrix, used to compute up check potential for box 1
for i in range(0,p):
  for j in range(0,N1):
    K1[i,j] = Laplace2D(rc1[i],zc1[i],rs1[j],zs1[j])

# green's function at rt2,zt2 on the upward check surface and rs2,zs2 sources
# p x N2 matrix, used to find up check potential for box 2
for i in range(0,p):
  for j in range(0,N2):
    K2[i,j] = Laplace2D(rc2[i],zc2[i],rs2[j],zs2[j])

# green's function at rt1,zt1 on up check surf & rq1,zq1 on up equiv surface
# p x p matrix, used to solve integral eqn for up equiv density of box 1
for i in range(0,p):
  for j in range(0,p):
    K3[i,j] = Laplace2D(rc1[i],zc1[i],rq1[j],zq1[j])

# green's function at rt2,zt2 on up check surf & rq2,zq2 on up equiv surface
# p x p matrix, used to solve integral eqn for up equiv density of box 2
for i in range(0,p):
  for j in range(0,p):
    K4[i,j] = Laplace2D(rc2[i],zc2[i],rq2[j],zq2[j])

# green's function at rq1,zq1 on down check surf & rq2,zq2 on up equiv surface
# p x p matrix, used to find down check potential for box 1
for i in range(0,p):
  for j in range(0,p):
    K5[i,j] = Laplace2D(rq1[i],zq1[i],rq2[j],zq2[j])

# green's function at rq1,zq1 on down check surf & rt1,zt1 on down equiv surf
# p x p matrix, used to solve integral eqn for down equiv density of box 1
for i in range(0,p):
  for j in range(0,p):
    K6[i,j] = Laplace2D(rq1[i],zq1[i],rc1[j],zc1[j])

# now solve for the upward check potentials
# q = K*phi
q1u = np.dot(K1,phi1)
q2u = np.dot(K2,phi2)

# now use Tikhonov regularization
# regularization parameter
alpha = np.power(10,-12)
# identity matrix
I = np.matlib.identity(p)
# solve the integral eq'n for the upward equivalent densities
eqd1u = np.dot(tikh(K3,p),q1u)

eqd2u = np.dot(tikh(K4,p),q2u)

# now for the M2L translation operator
# if box 2 is in the far field of box 1
# that is, use box 2 upward equiv density to find
# the downward equivalent density of box 1

# compute the downward check potential for box 1
q1d = np.dot(K5,eqd2u)

# then compute the downward equivalent density again using Tikhonov
eqd1d = np.dot(tikh(K6,p),q1d)

# I glossed over our use of it, but the M2L operator is a matrix:
# (p/(2*math.pi*radius1))*[(alpha*I + K6^T*K6)^(-1)]*K6^T*K5

#M2L = np.dot(tikh(K6,p),K5)
#print(M2L)