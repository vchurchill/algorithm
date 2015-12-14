# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import math
import matplotlib.pyplot as plt
from functions import *

# N is the number of sources in each box
N1 = 20

# p is the number of discretization points on the surfaces
p = 10

# 2*rb is the side length of each box
# for now, the box sizes are the same
rb = 0.5

# x,y is the position of the bottom left hand corner of the box
# change these to get different boxes
x1 = 0
y1 = 0

# c is the center of each box
c1 = [rb + x1, rb + y1]

# create sources outside box 1
sources1 = np.random.rand(N1,2)
sources1 = sources1 + [3, 3]

# create point inside box 1
point = [0.25,0.75]

# create some density for each source. Ying says this is "given"
# note: I wasn't sure what to put as these values
# just did random integers from 0 to 10
phi1 = np.random.randint(10, size=(N1,1))

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

# initiate kernel matrices
K1 = np.zeros(shape=(p,N1))
K2 = np.zeros(shape=(N1,1))
K3 = np.zeros(shape=(p,p))
K4 = np.zeros(shape=(p,1))

# initialize downward check potential vectors
q1d = np.zeros(shape=(p,1))

# break down source points into source coordinate vectors
for i in range(0,N1):
  rs1[i] = sources1[i,0]
  zs1[i] = sources1[i,1]

# create discretization points on surfaces
# define surface constant
d=0.1
# radii for surfaces
radius1 = (4-math.sqrt(2)-2*d)*rb
radius2 = (math.sqrt(2)+d)*rb

# box 1 downward equivalent surface points
for i in range(0,p):
  rc1[i] = radius1 * math.cos(math.pi*2 * i/p) + c1[0]
  zc1[i] = radius1 * math.sin(math.pi*2 * i/p) + c1[1]

# box 1 downward check surface points
for i in range(0,p):
  rq1[i] = radius2 * math.cos(math.pi*2 * i/p) + c1[0]
  zq1[i] = radius2 * math.sin(math.pi*2 * i/p) + c1[1]

# check that this is actually points in a box with a circle around it
# sources
plt.scatter(rs1,zs1,color='black')
# downward check surfaces
plt.scatter(rq1,zq1,color='red')
# downward equivalent surfaces
plt.scatter(rc1,zc1,color='blue')
# point in box
plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()

# calculate kernel matrices

# green's function at rq1,zq1 on the downward check surface and rs1,zs1 sources
for i in range(0,p):
  for j in range(0,N1):
    K1[i,j] = Laplace2D(rq1[i],zq1[i],rs1[j],zs1[j])

# green's function at point and rs1,zs1 sources
for j in range(0,N1):
  K2[j] = Laplace2D(point[0],point[1],rs1[j],zs1[j])

# green's function at rc1,zc1 on downward equivalent surf & rq1,zq1 on downward check surface
for i in range(0,p):
  for j in range(0,p):
    K3[i,j] = Laplace2D(rc1[i],zc1[i],rq1[j],zq1[j])

# green's function at rc1,zc1 on down equivalent surface & point
for i in range(0,p):
  K4[i] = Laplace2D(point[0],point[1],rc1[i],zc1[i])

# now solve for the downward check potentials
q1d = np.dot(K1,phi1)

# then compute the downward equivalent density again using Tikhonov
eqd1d = np.dot(tikh(K3,p),q1d)

potential = np.sum(np.multiply(K2,phi1))
estimate = np.sum(np.multiply(K4,eqd1d))
print("Actual:")
print(potential)
print("Estimate:")
print(estimate)

