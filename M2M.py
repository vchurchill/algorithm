# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
from green import *
import math
import matplotlib.pyplot as plt

# define tikhonov regularization function
def tikh(M,p):
  # regularization parameter
  alpha = np.power(10,-12)
  # identity matrix
  I = np.matlib.identity(p)
  return np.dot(np.linalg.inv(alpha*I+np.dot(np.matrix.transpose(M),M)),np.matrix.transpose(M))

def M2M(x1,y1,x2,y2,rb,p):
  # 2*rb is the side length of the child box
  # 4*rb is the side length of the parent box

  # p is the number of discretization points on the surfaces

  # x,y is the position of the bottom left hand corner of the box
  # change these to get different boxes

  # c is the center of each box
  c1 = [2*rb + x1, 2*rb + y1]
  c2 = [rb + x2, rb + y2]

  # initialize coordinate vectors
  # box 1 check surface
  rc1 = np.zeros(shape=(p,1))
  zc1 = np.zeros(shape=(p,1))
  # box 1 equiv surface
  rq1 = np.zeros(shape=(p,1))
  zq1 = np.zeros(shape=(p,1))
  # box 2 check surface
  rc2 = np.zeros(shape=(p,1))
  zc2 = np.zeros(shape=(p,1))
  # box 2 equiv surface
  rq2 = np.zeros(shape=(p,1))
  zq2 = np.zeros(shape=(p,1))

  # initiate kernel matrices
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
    rc1[i] = 2*radius1 * math.cos(math.pi*2 * i/p) + c1[0]
    zc1[i] = 2*radius1 * math.sin(math.pi*2 * i/p) + c1[1]

  # box 1 upward equivalent surface and downward check surface points
  for i in range(0,p):
    rq1[i] = 2*radius2 * math.cos(math.pi*2 * i/p) + c1[0]
    zq1[i] = 2*radius2 * math.sin(math.pi*2 * i/p) + c1[1]

  # box 2 upward check surface and downward equivalent surface points
  for i in range(0,p):
    rc2[i] = radius1 * math.cos(math.pi*2 * i/p) + c2[0]
    zc2[i] = radius1 * math.sin(math.pi*2 * i/p) + c2[1]

  # box 2 upward equivalent surface and downward check surface points
  for i in range(0,p):
    rq2[i] = radius2 * math.cos(math.pi*2 * i/p) + c2[0]
    zq2[i] = radius2 * math.sin(math.pi*2 * i/p) + c2[1]

  # upward equivalent surfaces
  plt.scatter(rq1,zq1,color='red')
  plt.scatter(rq2,zq2,color='purple')
  # upward check surfaces
  plt.scatter(rc1,zc1,color='blue')
  plt.scatter(rc2,zc2,color='green')
  plt.show()

  # calculate kernel matrices

  # green's function at rc1,zc1 on up check surf & rq2,zq2 on up equiv surface
  for i in range(0,p):
    for j in range(0,p):
      K1[i,j] = Laplace2D(rc1[i],zc1[i],rq2[j],zq2[j])

  # green's function at rq1,zq1 on up equiv surf & rc1,zc1 on up check surf
  for i in range(0,p):
    for j in range(0,p):
      K2[i,j] = Laplace2D(rq1[i],zq1[i],rc1[j],zc1[j])

  # now use Tikhonov regularization
  # regularization parameter
  alpha = np.power(10,-12)
  # identity matrix
  I = np.matlib.identity(p)
  # This matrix translates the upward equivalent density of a box 2
  # to the upward equivalent density of its parent box 1
  return np.dot(tikh(K2,p),K1)

print(M2M(0,0,0,0,1/2,16))
