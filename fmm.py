# import necessary packages
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt


# number of sources
n = 20

# create sources in box
sources = np.random.rand(n,2)

# initialize coordinate matrices with zeros
rs = np.zeros(shape=(n,1))
zs = np.zeros(shape=(n,1))

# separate coordinates of source points
for i in range(0,n):
  r = sources[i,0]
  z = sources[i,1]
  rs[i] = r
  zs[i] = z

# create targets on check surface
# radius
radius = 1.25
# numbers of target points to generate
m = 20
rt = [
  (radius * math.cos(t)+ 0.5)
  for t in (math.pi*2 * i/m for i in range(m))
]
zt = [
  (radius * math.sin(t)+0.5)
  for t in (math.pi*2 * i/m for i in range(m))
]

K = np.zeros(shape=(n,m))

for j in range(0,m):
  for i in range(0,n):
    K[i,j] = np.log(math.sqrt((rs[i] - rt[j])*(rs[i]-rt[j])
    + (zs[i]-zt[j])*(zs[i]-zt[j])))

print(K)
