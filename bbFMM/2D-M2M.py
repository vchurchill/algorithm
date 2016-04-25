# import necessary packages
import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

''' define Chebyshev functions and log kernel '''

# Chebyshev polynomial function
def T(n,x,a,b):
  return np.cos(n*np.arccos(((2/(b-a))*(x-((a+b)/2)))))

# compute Chebyshev nodes in interval -1,1
def nodes(n,a,b):
  nodes = np.zeros(shape=(n,1))
  for i in range(0,n):
    nodes[i] = (a+b)/2+ ((b-a)/2)*np.cos(((2*(i+1)-1)*np.pi)/(2*n))
  return nodes

# define other Chebyshev polynomial function
def S(n,x,y,a,b):
  k=1
  sum=0
  while k <= (n-1):
    sum+=T(k,x,a,b)*T(k,y,a,b)
    k+=1
  return 1/n + (2/n)*sum

# define multivariable version of above S function
def R(n,r1,z1,r2,z2,a,b,c,d):
  return S(n,r1,r2,a,b)*S(n,z1,z2,c,d)

# define log kernel
def log(r1,z1,r2,z2):
  return np.log(np.sqrt(np.square(r1-r2)+np.square(z1-z2)))

''' specify source number and node number '''

# N is the number of sources in each interval
N = 10

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=8

''' specify source interval '''
# source parent
a1 = -4
b1 = -2
a2 = -1
b2 = 1

# source children horizontal
C1a1 = a1
C1b1 = (b1+a1)/2
C2a1 = (b1+a1)/2
C2b1 = b1

# source children vertical
C1a2 = a2
C1b2 = (a2+b2)/2
C2a2 = (a2+b2)/2
C2b2 = b2

# target
c1 = 3
d1 = 5
c2 = 3
d2 = 5

''' populate boxes '''

# create sources in child boxes
sources1 = np.random.rand(N,2)
sources2 = np.random.rand(N,2)
sources3 = np.random.rand(N,2)
sources4 = np.random.rand(N,2)
for i in range(0,N):
  sources1[i,0] = sources1[i,0]*(C1b1-C1a1)+C1a1
  sources1[i,1] = sources1[i,1]*(C1b2-C1a2)+C1a2
for i in range(0,N):
  sources2[i,0] = sources2[i,0]*(C1b1-C1a1)+C1a1
  sources2[i,1] = sources2[i,1]*(C2b2-C2a2)+C2a2
for i in range(0,N):
  sources3[i,0] = sources3[i,0]*(C2b1-C2a1)+C2a1
  sources3[i,1] = sources3[i,1]*(C1b2-C1a2)+C1a2
for i in range(0,N):
  sources4[i,0] = sources4[i,0]*(C2b1-C2a1)+C2a1
  sources4[i,1] = sources4[i,1]*(C2b2-C2a2)+C2a2

# create target
point = [4.1,4.3]

'''
for i in range(0,N):
  plt.scatter(sources1[i,0],sources1[i,1],color='black')
  plt.scatter(sources2[i,0],sources2[i,1],color='red')
  plt.scatter(sources3[i,0],sources3[i,1],color='blue')
  plt.scatter(sources4[i,0],sources4[i,1],color='yellow')
  plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()
'''

# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigma1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma1[i] = (np.random.randint(0,2)*2)-1

sigma2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma2[i] = (np.random.randint(0,2)*2)-1

sigma3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma3[i] = (np.random.randint(0,2)*2)-1

sigma4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigma4[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each child box '''

# each Chebyshev node, m, gets a weight
def W1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C1a1,C1b1)[m1],nodes(n,C1a2,C1b2)[m2],sources1[j,0],sources1[j,1],C1a1,C1b1,C1a2,C1b2)*sigma1[j]
  return sum

def W2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C1a1,C1b1)[m1],nodes(n,C2a2,C2b2)[m2],sources2[j,0],sources2[j,1],C1a1,C1b1,C2a2,C2b2)*sigma2[j]
  return sum

def W3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C2a1,C2b1)[m1],nodes(n,C1a2,C1b2)[m2],sources3[j,0],sources3[j,1],C2a1,C2b1,C1a2,C1b2)*sigma3[j]
  return sum

def W4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C2a1,C2b1)[m1],nodes(n,C2a2,C2b2)[m2],sources4[j,0],sources4[j,1],C2a1,C2b1,C2a2,C2b2)*sigma4[j]
  return sum

# PARENT
def W(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources1[j,0],sources1[j,1],a1,b1,a2,b2)*sigma1[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources2[j,0],sources2[j,1],a1,b1,a2,b2)*sigma2[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources3[j,0],sources3[j,1],a1,b1,a2,b2)*sigma3[j] + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],sources4[j,0],sources4[j,1],a1,b1,a2,b2)*sigma4[j]
  return sum

''' this is the M2M operation taking the W weights of the child boxes and translating to the parent '''
def West(m1,m2):
  sum = 0
  for mprime1 in range(0,n):
    for mprime2 in range(0,n):
      sum += R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C1a1,C1b1)[mprime1],nodes(n,C1a2,C1b2)[mprime2],a1,b1,a2,b2)*W1(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C1a1,C1b1)[mprime1],nodes(n,C2a2,C2b2)[mprime2],a1,b1,a2,b2)*W2(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C2a1,C2b1)[mprime1],nodes(n,C1a2,C1b2)[mprime2],a1,b1,a2,b2)*W3(mprime1,mprime2) + R(n,nodes(n,a1,b1)[m1],nodes(n,a2,b2)[m2],nodes(n,C2a1,C2b1)[mprime1],nodes(n,C2a2,C2b2)[mprime2],a1,b1,a2,b2)*W4(mprime1,mprime2)
  return sum

print("weight from parent box calculated by itself")
for m1 in range(0,n):
  for m2 in range(0,n):
    print(W(m1,m2))

print("weight from parent box calculated from child boxes and M2M")
for m1 in range(0,n):
  for m2 in range(0,n):
    print(West(m1,m2))