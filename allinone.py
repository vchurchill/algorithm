# this code computes the M2L operator between two boxes of the same size

# import necessary packages
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

# N is number of sources in each box
N1 = 20
N2 = 40

# p is the number of discretization points
p = 10

# 2*rb is the side length of each box
rb = 0.5

# r,z is the position of the bottom left hand corner of the box
x1 = 0
y1 = 0
x2 = 3
y2 = 0

# c is the center of the box
c1 = [rb + x1, rb + y1]
c2 = [rb + x2, rb + y2]
  
# define constant about the check surface
d=0.1
  
# create sources for box 1
sources1 = np.random.rand(N1,2)
sources1 = sources1 + [x1, y1]

# create sources for box 2
sources2 = np.random.rand(N2,2)
sources2 = sources2 + [x2, y2]

# create a density for each source, phi
phi1 = np.random.randint(10, size=(N1,1))
phi2 = np.random.randint(10, size=(N2,1))

# initialize coordinate matrices with zeros
rs1 = np.zeros(shape=(N1,1))
zs1 = np.zeros(shape=(N1,1))
rt1 = np.zeros(shape=(p,1))
zt1 = np.zeros(shape=(p,1))

rs2 = np.zeros(shape=(N2,1))
zs2 = np.zeros(shape=(N2,1))
rt2 = np.zeros(shape=(p,1))
zt2 = np.zeros(shape=(p,1))
  
# separate coordinates of source points
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

# check surface radius (slightly less than 1.2 for a box of sidelength 1)
radius = (4-math.sqrt(2)-2*d)*rb
print(radius)

# create points on check surfaces
for i in range(0,p):
  rt1[i] = radius * math.cos(math.pi*2 * i/p) + c1[0]
  zt1[i] = radius * math.sin(math.pi*2 * i/p) + c1[1]

for i in range(0,p):
  rt2[i] = radius * math.cos(math.pi*2 * i/p) + c2[0]
  zt2[i] = radius * math.sin(math.pi*2 * i/p) + c2[1]

# check that this is actually points in a box with a circle around it
plt.scatter(rs1,zs1)
plt.scatter(rs2,zs2)
plt.scatter(rt1,zt1)
plt.scatter(rt2,zt2)
plt.show()
  
#print(rs)
#print(zs)
#print(rt)
#print(zt)
  
# initiate kernel matrix
K1 = np.zeros(shape=(p,N1))
K2 = np.zeros(shape=(p,N2))

# populate kernel matrices
for i in range(0,p):
  for j in range(0,N1):
    K1[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rs1[j]-rt1[i])+
    np.square(zs1[j]-zt1[i])))

for i in range(0,p):
  for j in range(0,N2):
    K2[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rs2[j]-rt2[i])+
    np.square(zs2[j]-zt2[i])))

#print(K)
#print(phi)
  
# initiate check potential vector
q1 = np.zeros(shape=(p,1))
q2 = np.zeros(shape=(p,1))

# solve for the check potential
for i in range(0,p):
  for j in range(0,N1):
    q1[i] = q1[i]+ K1[i,j]*phi1[j]

for i in range(0,p):
  for j in range(0,N2):
    q2[i] = q2[i]+ K2[i,j]*phi2[j]
  
print(q1)
print(q2)