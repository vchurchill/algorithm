# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import scipy
from scipy import special
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

def M2L(x1,y1,x2,y2,rb,p):
  # this is the M2L from box 2 to box 1

  # x,y is the position of the bottom left hand corner of the boxes
  # change these to get different boxes 1 and 2

  # 2*rb is the side length of each box
  # for now, the box sizes are the same

  # p is the number of discretization points on the surfaces

  # box 1 check surface
  rc1 = np.zeros(shape=(p,1))
  zc1 = np.zeros(shape=(p,1))
  # box 1 equiv surface
  rq1 = np.zeros(shape=(p,1))
  zq1 = np.zeros(shape=(p,1))
  # box 2 equiv surface
  rq2 = np.zeros(shape=(p,1))
  zq2 = np.zeros(shape=(p,1))
  # kernel matrices
  K1 = np.zeros(shape=(p,p))
  K2 = np.zeros(shape=(p,p))

  # c is the center of each box
  c1 = [rb + x1, rb + y1]
  c2 = [rb + x2, rb + y2]
  
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

  # calculate kernel matrices
  # green's function at rq1,zq1 on down check surf & rq2,zq2 on up equiv surf
  # p x p matrix, used to find down check potential for box 1
  for i in range(0,p):
    for j in range(0,p):
      K1[i,j] = Laplace3D(rq1[i],zq1[i],rq2[j],zq2[j])

  # green's function at rq1,zq1 on down check surf & rc1,zc1 on down equiv surf
  # p x p matrix, used to solve integral eqn for down equiv density of box 1
  for i in range(0,p):
    for j in range(0,p):
      K2[i,j] = Laplace3D(rq1[i],zq1[i],rc1[j],zc1[j])

  # The M2L operator translating up equiv density of box 2
  # to down equiv density of box 1 is this matrix:

  return np.dot(tikh(K2,p),K1)


#import sys
#sys.stdout = open("/Users/HomeBase/Github/algorithm/output.txt","w")
print(M2L(0,0,0.5,0,.125,7))
