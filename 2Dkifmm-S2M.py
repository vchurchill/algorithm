# import necessary packages
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

def checkpotential(N,p,rb):
  # N is number of sources, p is number of discretization points
  # on the equivalent and check surfaces
  # length of box side is 2r
  
  # define constant about the check surface
  d=0.1
  
  # create sources in box, y
  sources = np.random.rand(N,2)

  # create a density for each source, phi
  phi = np.random.randint(10, size=(N,1))

  # initialize coordinate matrices with zeros
  rs = np.zeros(shape=(N,1))
  zs = np.zeros(shape=(N,1))
  rt = np.zeros(shape=(p,1))
  zt = np.zeros(shape=(p,1))
  
  # separate coordinates of source points
  for i in range(0,N):
    r = sources[i,0]
    z = sources[i,1]
    rs[i] = r
    zs[i] = z

  # check surface radius
  radius = (4-math.sqrt(2)-2*d)*rb
  #print(radius)

  # create targets on check surface
  for i in range(0,p):
      rt[i] = radius * math.cos(math.pi*2 * i/p) + rb
      zt[i] = radius * math.sin(math.pi*2 * i/p) + rb

  # check that this is actually points in a box with a circle around it
  #plt.scatter(rs,zs)
  #plt.scatter(rt,zt)
  #plt.show()
  
  #print(rs)
  #print(zs)
  #print(rt)
  #print(zt)
  
  # initiate kernel matrix
  K = np.zeros(shape=(p,N))

  # populate kernel matrix
  for i in range(0,p):
    for j in range(0,N):
      K[i,j] = (1/(2*math.pi))*np.log(math.sqrt(np.square(rs[j]-rt[i])+
      np.square(zs[j]-zt[i])))

  #print(K)
  #print(phi)
  
  # initiate check potential vector
  q = np.zeros(shape=(p,1))
  
  # solve for the check potential
  for i in range(0,p):
    for j in range(0,N):
      q[i] = q[i]+ K[i,j]*phi[j]
  
  print(q)

checkpotential(20,10,0.5)