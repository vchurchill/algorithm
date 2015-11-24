# import necessary packages
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt

# create sources in box
sources = np.random.rand(20,3)

# initialize other matrices with zeros
xs = np.zeros(shape=(20,1))
ys = np.zeros(shape=(20,1))
zs = np.zeros(shape=(20,1))
r = np.zeros(shape=(20,1))
theta = np.zeros(shape=(20,1))
csources = np.zeros(shape=(20,3))

# 
for i in range(0,20):
  x = sources[i,0]
  y = sources[i,1]
  z = sources[i,2]
  xs[i] = x
  ys[i] = y
  zs[i] = z
  r[i] = np.sqrt(m.pow(x,2) + m.pow(y,2))
  theta[i] = np.arctan(y/x)
  csources[i,0] = r[i]
  csources[i,1] = theta[i]
  csources[i,2] = z

# create targets on 
rad = 1.25  #radius
n = 20 #points to generate
xt = [
  (rad * m.cos(t)+ 0.5)
  for t in (m.pi*2 * i/n for i in range(n))
]
yt = [
  (rad * m.sin(t)+0.5)
  for t in (m.pi*2 * i/n for i in range(n))
]

# plt.scatter(xs,ys)
# plt.scatter(xt,yt)
# plt.show()

f = np.zeros(shape=(20,1))

for i in range (0,20):
  f[i] = np.log(r[i]*r[i] + xt[i]*xt[i] 
  - 2*r[i]*xt[i]*m.cos(theta[i]) + zs[i]*zs[i])

print(f)
