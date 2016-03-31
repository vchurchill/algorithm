# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
from functions import *
import math
import matplotlib.pyplot as plt

def L2L(x1,y1,x2,y2,rb,p):
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

  # box 1 downward equivalent surface points
  for i in range(0,p):
    rc1[i] = 2*radius1 * math.cos(math.pi*2 * i/p) + c1[0]
    zc1[i] = 2*radius1 * math.sin(math.pi*2 * i/p) + c1[1]

  # box 1 downward check surface points
  for i in range(0,p):
    rq1[i] = 2*radius2 * math.cos(math.pi*2 * i/p) + c1[0]
    zq1[i] = 2*radius2 * math.sin(math.pi*2 * i/p) + c1[1]

  # box 2 downward equivalent surface points
  for i in range(0,p):
    rc2[i] = radius1 * math.cos(math.pi*2 * i/p) + c2[0]
    zc2[i] = radius1 * math.sin(math.pi*2 * i/p) + c2[1]

  # box 2 downward check surface points
  for i in range(0,p):
    rq2[i] = radius2 * math.cos(math.pi*2 * i/p) + c2[0]
    zq2[i] = radius2 * math.sin(math.pi*2 * i/p) + c2[1]

  # down check surfaces
  #plt.scatter(rq1,zq1,color='blue')
  #plt.scatter(rq2,zq2,color='blue')
  # down equiv surfaces
  #plt.scatter(rc1,zc1,color='red')
  #plt.scatter(rc2,zc2,color='red')
  #plt.show()

  # calculate kernel matrices
  # truncation parameter
  n=3
  # green's function at rc1,zc1 on down equiv surf & rq2,zq2 on down check surface
  for i in range(0,p):
    for j in range(0,p):
      K1[i,j] = LaplaceMode(rq2[i],zq2[i],rc1[j],zc1[j],n)

  # green's function at rq1,zq1 on down check surf & rc1,zc1 on down equiv surf
  for i in range(0,p):
    for j in range(0,p):
      K2[i,j] = LaplaceMode(rq2[i],zq2[i],rc2[j],zc2[j],n)

  print(K1)
  print(K2)

  return np.dot(np.linalg.inv(K2),K1)

print(L2L(4,4,4,4,1,10))
