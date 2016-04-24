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

# define log kernel
def log(r,z):
  return np.log(np.fabs(r-z))

''' specify source number and node number '''

# N is the number of sources in each interval
N = 5

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=10

''' specify source interval '''
# source parent
a = -4
b = -2

# source children
C1a = a
C1b = a + (b-a)/4
C2a = a + (b-a)/4
C2b = (a+b)/2
C3a = (a+b)/2
C3b = a + 3*(b-a)/4
C4a = a + 3*(b-a)/4
C4b = b

# target
c = 3
d = 5

''' populate boxes '''

# create sources in child boxes separately
sources1 = np.random.rand(N,1)*(C1b-C1a) + C1a
sources2 = np.random.rand(N,1)*(C2b-C2a) + C2a
sources3 = np.random.rand(N,1)*(C3b-C3a) + C3a
sources4 = np.random.rand(N,1)*(C4b-C4a) + C4a

# create target
point = 4.5


sourcesy = np.zeros(shape=(N,1))
plt.scatter(sources1,sourcesy,color='black')
plt.scatter(sources2,sourcesy,color='red')
plt.scatter(sources3,sourcesy,color='blue')
plt.scatter(sources4,sourcesy,color='yellow')
plt.scatter(point,0,color='green')
plt.grid()
plt.show()


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

print(sigma1)
print(sigma2)
print(sigma3)
print(sigma4)
''' these are the weights for each child box '''

# each Chebyshev node, m, gets a weight
def W1(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources1[j],a,b)*sigma1[j]
  return sum

def W2(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources2[j],a,b)*sigma2[j]
  return sum
  
def W3(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources3[j],a,b)*sigma3[j]
  return sum

def W4(m,a,b):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources4[j],a,b)*sigma4[j]
  return sum

# PARENT
def W(m):
  sum = 0
  for j in range(0,N):
    sum += S(n,nodes(n,a,b)[m],sources1[j],a,b)*sigma1[j] + S(n,nodes(n,a,b)[m],sources2[j],a,b)*sigma2[j] + S(n,nodes(n,a,b)[m],sources3[j],a,b)*sigma3[j] + S(n,nodes(n,a,b)[m],sources4[j],a,b)*sigma4[j]
  return sum

''' this is the M2M operation taking the W weights of the child boxes and translating to the parent '''
def West(m):
  sum = 0
  for mprime in range(0,n):
    sum += S(n,nodes(n,a,b)[m],nodes(n,C1a,C1b)[mprime],a,b)*W1(mprime,C1a,C1b) + S(n,nodes(n,a,b)[m],nodes(n,C2a,C2b)[mprime],a,b)*W2(mprime,C2a,C2b) + S(n,nodes(n,a,b)[m],nodes(n,C3a,C3b)[mprime],a,b)*W3(mprime,C3a,C3b) + S(n,nodes(n,a,b)[m],nodes(n,C4a,C4b)[mprime],a,b)*W4(mprime,C4a,C4b)
  return sum

print("weight from parent box calculated by itself")
wparent = 0
for m in range(0,n):
  wparent += W(m)
print(wparent)

for m in range(0,n):
  print(W(m))

print("weight from parent box calculated from child boxes and M2M")
wmtm = 0
for m in range(0,n):
  wmtm += West(m)
print(wmtm)

for m in range(0,n):
  print(West(m))