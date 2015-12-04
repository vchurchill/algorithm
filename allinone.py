# this code computes the M2L operator between two boxes of the same size

# import necessary packages
import numpy as np
import numpy.matlib
import numpy.linalg
import scipy as sp
import math
import matplotlib.pyplot as plt

# N is number of sources in each box
N1 = 400
N2 = 400

# p is the number of discretization points on the surfaces
p = 16

# 2*rb is the side length of each box
rb = 0.5

# x,y is the position of the bottom left hand corner of the box
x1 = 0
y1 = 0
x2 = 3
y2 = 0

# c is the center of the box
c1 = [rb + x1, rb + y1]
c2 = [rb + x2, rb + y2]

# define surface constant
d=0.1

# create sources for box 1
sources1 = np.random.rand(N1,2)
sources1 = sources1 + [x1, y1]

# create sources for box 2
sources2 = np.random.rand(N2,2)
sources2 = sources2 + [x2, y2]

# create some density for each source, phi. ying says this is "given"
phi1 = np.random.randint(10, size=(N1,1))
phi2 = np.random.randint(10, size=(N2,1))

# initialize source coordinate matrices with zeros, otherwise errors
# box 1 sources
rs1 = np.zeros(shape=(N1,1))
zs1 = np.zeros(shape=(N1,1))
# box 1 check surface
rt1 = np.zeros(shape=(p,1))
zt1 = np.zeros(shape=(p,1))
# box 1 equiv surface
rq1 = np.zeros(shape=(p,1))
zq1 = np.zeros(shape=(p,1))
# box 2 sources
rs2 = np.zeros(shape=(N2,1))
zs2 = np.zeros(shape=(N2,1))
# box 2 check surface
rt2 = np.zeros(shape=(p,1))
zt2 = np.zeros(shape=(p,1))
# box 3 equiv surface
rq2 = np.zeros(shape=(p,1))
zq2 = np.zeros(shape=(p,1))
  
# break down source coordinates
for i in range(0,N1):
  r1 = sources1[i,0]
  z1 = sources1[i,1]
  rs1[i] = r1
  zs1[i] = z1
  
for i in range(0,N2):
  r2 = sources2[i,0]
  z2 = sources2[i,1]
  rs2[i] = r2
  zs2[i] = z2

# radii for surfaces
radius = (4-math.sqrt(2)-2*d)*rb
radiusd = (math.sqrt(2)+d)*rb

# box 1 upward check surface and downward equivalent surface
for i in range(0,p):
  rt1[i] = radius * math.cos(math.pi*2 * i/p) + c1[0]
  zt1[i] = radius * math.sin(math.pi*2 * i/p) + c1[1]

# box 1 upward equivalent surface and downward check surface
for i in range(0,p):
  rq1[i] = radiusd * math.cos(math.pi*2 * i/p) + c1[0]
  zq1[i] = radiusd * math.sin(math.pi*2 * i/p) + c1[1]

# box 2 upward check surface and downward equivalent surface
for i in range(0,p):
  rt2[i] = radius * math.cos(math.pi*2 * i/p) + c2[0]
  zt2[i] = radius * math.sin(math.pi*2 * i/p) + c2[1]

# box 2 upward equivalent surface and downward check surface
for i in range(0,p):
  rq2[i] = radiusd * math.cos(math.pi*2 * i/p) + c2[0]
  zq2[i] = radiusd * math.sin(math.pi*2 * i/p) + c2[1]

# check that this is actually points in a box with a circle around it
plt.scatter(rs1,zs1)
plt.scatter(rs2,zs2)
plt.scatter(rt1,zt1)
plt.scatter(rt2,zt2)
#plt.show()
# notice if you plot this, the upward check surfaces for box 1 and 2 overlap
# the radii are prescribed by Ying, which does not restrict overlapping
# check surfaces
  
#print(rs)
#print(zs)
#print(rt)
#print(zt)
  
# initiate kernel matrices
K1 = np.zeros(shape=(p,N1))
K2 = np.zeros(shape=(p,N2))
K3 = np.zeros(shape=(p,p))
K4 = np.zeros(shape=(p,p))
K5 = np.zeros(shape=(p,p))
K6 = np.zeros(shape=(p,p))

# calculate kernel matrices
# green's function at rt1,zt1 on the upward check surface and rs1,zs1 sources
for i in range(0,p):
  for j in range(0,N1):
    K1[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rt1[i]-rs1[j])+
    np.square(zt1[i]-zs1[j])))

# green's function at rt2,zt2 on the upward check surface and rs2,zs2 sources
for i in range(0,p):
  for j in range(0,N2):
    K2[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rt2[i]-rs2[j])+
    np.square(zt2[i]-zs2[j])))

# green's function at rt1,zt1 on up check surf & rq1,zq1 on up equiv surface
for i in range(0,p):
  for j in range(0,p):
    K3[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rt1[i]-rq1[j])+
    np.square(zt1[i]-zq1[j])))

# green's function at rt2,zt2 on up check surf & rq2,zq2 on up equiv surface
for i in range(0,p):
  for j in range(0,p):
    K4[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rt2[i]-rq2[j])+
    np.square(zt2[i]-zq2[j])))

# green's function at rq1,zq1 on down check surf & rq2,zq2 on up equiv surface
for i in range(0,p):
  for j in range(0,p):
    K5[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rq1[i]-rq2[j])+
    np.square(zq1[i]-zq2[j])))

# green's function at rq1,zq1 on down check surf & rt1,zt1 on down equiv surface
for i in range(0,p):
  for j in range(0,p):
    K6[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rq1[i]-rt1[j])+
    np.square(zq1[i]-zt1[j])))


#print(K)
#print(phi)
  
# initiate upward check potential vectors
q1u = np.zeros(shape=(p,1))
q2u = np.zeros(shape=(p,1))

# solve for the upward check potentials
q1u = np.dot(K1,phi1)
q2u = np.dot(K2,phi2)

#for i in range(0,p):
#  for j in range(0,N1):
#    q1[i] = q1[i]+ K1[i,j]*phi1[j]

#for i in range(0,p):
#  for j in range(0,N2):
#    q2[i] = q2[i]+ K2[i,j]*phi2[j]

# solve the integral eq'n for the upward equivalent density using tikhonov
# tikhonov regularization parameter
#alpha = np.power(10, -12)
#I1 = np.matlib.identity(N1)
#I2 = np.matlib.identity(N2)

eqd1u = p*np.dot(np.linalg.inv(K3),q1u)/(2*math.pi*radiusd)
eqd2u = p*np.dot(np.linalg.inv(K4),q2u)/(2*math.pi*radiusd)

#eqd1u = np.dot(np.dot(np.linalg.inv(alpha * I1 +
#np.dot(np.matrix.transpose(K3),K3)),np.matrix.transpose(K3)),q1)

#eqd2u = np.dot(np.dot(np.linalg.inv(alpha * I2 +
#np.dot(np.matrix.transpose(K4),K4)),np.matrix.transpose(K4)),q2)

#print(eqd1)
#print(eqd2)

# for the M2L translation operator if box 2 is in the far field of box 1
# that is, the downward equivalent density of box 1 using info from box 2

# initialize downward check potential vectors
q1d = np.zeros(shape=(p,1))

# compute the downward check potential for box 1
q1d = np.dot(K5,eqd2u)

# then compute the downward equivalent density
eqd1d = p*np.dot(np.linalg.inv(K6),q1d)/(2*math.pi*radius)

print(eqd1d)
