# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' define Chebyshev functions '''

# Chebyshev polynomial function
def T(n,x):
  return np.cos(n*np.arccos(x))

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

''' specify source number and node number '''

# N is the number of sources in each box
N = 15

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' specify source box and target box '''

a = 0
b = 1
c = 0
d = 1
e = 2
f = 3
g = 2
h = 3

''' populate boxes '''

# create sources in box
sources = np.random.rand(N,2) + [a,c]
print(sources)

# create target
point = [2.6,2.3]

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

''' define some R vectors/matrices with nodes and sources '''

# define P vector of R between nodes in source box and sources
def P(m1,m2):
  P = np.zeros(shape=(N,1))
  P[i] = R(n,nodes(n,a,b)[m1],nodes(n,c,d)[m2],rs[i],zs[i])
  return P

# define Q vector of R between nodes in target box and target point
def Q(l1,l2):
  Q = R(n,nodes(n,e,f)[l1],nodes(n,g,h)[l2],point[0],point[1])
  return Q

''' compute equivalent densities at source-box Chebyshev nodes '''

def W(m1,m2):
  W = np.dot(sigma.T,P(m1,m2))

''' sum of the multiplication elements of the kern matrices x weight matrices '''
''' this is #2 in the fast summation explanation '''
def fnode(l1,l2,k):
  for i in range(0,n):
    for j in range(0,n):
      fnode += W(i,j,k)*log(nodes(n,e,f)[l1],nodes(n,g,h)[l2],nodes(n,a,b)[i],nodes(n,c,d)[j])
  return fnode

''' now we need to compute f(x_i) by summing the multiplication of elements of fnode and Q '''
fest = 0
for k in range(0,N):
  for i in range(0,n):
    for j in range(0,n):
      fest += fnode(i,j,k)*Q(i,j)

''' now we find the actual potential on the target point due to the sources in the box '''

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
    plt.scatter(nodes(n,e,f)[i],nodes(n,g,h)[j],color='blue')

plt.grid()
#plt.show()