# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' function definitions '''

# Chebyshev polynomial function
def T(n,x):
  if n==0:
    return 1
  if n==1:
    return x
  else:
    return 2*x*T(n-1,x)-T(n-2,x)

# computes chebyshev nodes in interval a,b
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i]=(a+b)/2 + ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x)*T(k,y)
    k+=1
  return 1/n + (2/n)*sum

# define multivariable version of above S function
def R(n,r1,z1,r2,z2):
  return S(n,r1,r2)*S(n,z1,z2)

# define log kernel
def log(r1,z1,r2,z2):
  return np.log(np.sqrt(np.square(r1-r2)+np.square(z1-z2)))

# define P vector of R between nodes in source box and sources
def P(m1,m2,q,r,p,s):
  P = np.zeros(shape=(N,1))
  for i in range(0,n):
    P[i] = R(n,nodes(n,q,r)[m1],nodes(n,p,s)[m2],rs[i],zs[i])
  return P

''' specify source number and node number '''

# N is the number of sources in each box
N = 40

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n = 5

''' specify source box and target box '''

a = 0
b = 1
c = 0
d = 1
e = 1
f = 2
g = 1
h = 2

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,2) + [a,c]
print("These are the source points: ")
print(sources)

# create target
point = [1.6,1.3]

# create some charge (density) for each source
# they can be +1 or -1
sigma = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma[i] = (np.random.randint(0,2)*2)-1

# initialize source coordinate vectors
rs = np.zeros(shape=(N,1))
zs = np.zeros(shape=(N,1))

# break down source points into source coordinate vectors
for i in range(0,N):
  rs[i] = sources[i,0]
  zs[i] = sources[i,1]

''' compute equivalent densities at source-box Chebyshev nodes '''

W = np.zeros(shape=(n,n))
for i in range(0,n):
  for j in range(0,n):
    W[i,j] = np.dot(sigma.T,P(i,j,a,b,c,d))

print("W weight matrix:")
print(W)

''' compute kernel matrix between target point and source-box Chebyshev nodes ''' 

Kest = np.zeros(shape=(n,n))
for i in range(0,n):
  for j in range(0,n):
    Kest[i,j] = log(point[0],point[1],nodes(n,a,b)[i],nodes(n,c,d)[j])
print("Kernel estimate")
print(Kest)

''' compute sum of kernel matrix ij element multiplied by W ij element '''
''' this is the sum of the elements of the Hadamard product of Kest and W '''

fest = 0
for i in range(0,n):
  for j in range(0,n):
    fest += Kest[i,j]*W[i,j]

'''
now we find the actual potential due to the real source densities
'''

Kact= np.zeros(shape=(1,N))
for i in range(0,N):
  Kact[0,i] = log(point[0],point[1],rs[i],zs[i])

print("K actual")
print(Kact)

fact = np.dot(Kact,sigma)

print("Actual:")
print(fact)
print("Estimate:")
print(fest)

# check that this is actually points in a box with a circle around it
# sources
plt.scatter(rs,zs,color='black')
# point
plt.scatter(point[0],point[1],color='green')
# nodes
for i in range(0,n):
  for j in range(0,n):
    plt.scatter(nodes(n,a,b)[i],nodes(n,c,d)[j],color='red')

plt.grid()
#plt.show()