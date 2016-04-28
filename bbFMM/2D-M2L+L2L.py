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
N = 1

# n is the number of Chebyshev nodes in each interval
# n^2 interpolation points in each box
n=5

''' specify source interval '''
# parent box
Pa1 = 2
Pb1 = 4
Pa2 = 2
Pb2 = 4

# target box
a1 = 3
b1 = 4
a2 = 3
b2 = 4

# parent interaction list boxes
P1c1 = 0
P1d1 = 2
P1c2 = 6
P1d2 = 8

P2c1 = 2
P2d1 = 4
P2c2 = 6
P2d2 = 8

P3c1 = 4
P3d1 = 6
P3c2 = 6
P3d2 = 8

P4c1 = 6
P4d1 = 8
P4c2 = 6
P4d2 = 8

P5c1 = 6
P5d1 = 8
P5c2 = 4
P5d2 = 6

P6c1 = 6
P6d1 = 8
P6c2 = 2
P6d2 = 4

P7c1 = 6
P7d1 = 8
P7c2 = 0
P7d2 = 2

# target interaction list boxes
C1a1 = 0
C1b1 = 1
C1a2 = 0
C1b2 = 1

C2a1 = 0
C2b1 = 1
C2a2 = 1
C2b2 = 2

C3a1 = 0
C3b1 = 1
C3a2 = 2
C3b2 = 3

C4a1 = 0
C4b1 = 1
C4a2 = 3
C4b2 = 4

C5a1 = 0
C5b1 = 1
C5a2 = 4
C5b2 = 5

C6a1 = 0
C6b1 = 1
C6a2 = 5
C6b2 = 6

C7a1 = 1
C7b1 = 2
C7a2 = 0
C7b2 = 1

C8a1 = 1
C8b1 = 2
C8a2 = 1
C8b2 = 2

C9a1 = 1
C9b1 = 2
C9a2 = 2
C9b2 = 3

C10a1 = 1
C10b1 = 2
C10a2 = 3
C10b2 = 4

C11a1 = 1
C11b1 = 2
C11a2 = 4
C11b2 = 5

C12a1 = 1
C12b1 = 2
C12a2 = 5
C12b2 = 6

C13a1 = 5
C13b1 = 6
C13a2 = 0
C13b2 = 1

C14a1 = 5
C14b1 = 6
C14a2 = 1
C14b2 = 2

C15a1 = 5
C15b1 = 6
C15a2 = 2
C15b2 = 3

C16a1 = 5
C16b1 = 6
C16a2 = 3
C16b2 = 4

C17a1 = 5
C17b1 = 6
C17a2 = 4
C17b2 = 5

C18a1 = 5
C18b1 = 6
C18a2 = 5
C18b2 = 6

C19a1 = 2
C19b1 = 3
C19a2 = 0
C19b2 = 1

C20a1 = 2
C20b1 = 3
C20a2 = 1
C20b2 = 2

C21a1 = 2
C21b1 = 3
C21a2 = 5
C21b2 = 6

C22a1 = 3
C22b1 = 4
C22a2 = 0
C22b2 = 1

C23a1 = 3
C23b1 = 4
C23a2 = 1
C23b2 = 2

C24a1 = 3
C24b1 = 4
C24a2 = 5
C24b2 = 6

C25a1 = 4
C25b1 = 5
C25a2 = 0
C25b2 = 1

C26a1 = 4
C26b1 = 5
C26a2 = 1
C26b2 = 2

C27a1 = 4
C27b1 = 5
C27a2 = 5
C27b2 = 6

''' populate boxes '''

# create sources
sourcesP1 = np.random.rand((2*N),2)
sourcesP2 = np.random.rand((2*N),2)
sourcesP3 = np.random.rand((2*N),2)
sourcesP4 = np.random.rand((2*N),2)
sourcesP5 = np.random.rand((2*N),2)
sourcesP6 = np.random.rand((2*N),2)
sourcesP7 = np.random.rand((2*N),2)

sourcesC1 = np.random.rand(N,2)
sourcesC2 = np.random.rand(N,2)
sourcesC3 = np.random.rand(N,2)
sourcesC4 = np.random.rand(N,2)
sourcesC5 = np.random.rand(N,2)
sourcesC6 = np.random.rand(N,2)
sourcesC7 = np.random.rand(N,2)
sourcesC8 = np.random.rand(N,2)
sourcesC9 = np.random.rand(N,2)
sourcesC10 = np.random.rand(N,2)
sourcesC11 = np.random.rand(N,2)
sourcesC12 = np.random.rand(N,2)
sourcesC13 = np.random.rand(N,2)
sourcesC14 = np.random.rand(N,2)
sourcesC15 = np.random.rand(N,2)
sourcesC16 = np.random.rand(N,2)
sourcesC17 = np.random.rand(N,2)
sourcesC18 = np.random.rand(N,2)
sourcesC19 = np.random.rand(N,2)
sourcesC20 = np.random.rand(N,2)
sourcesC21 = np.random.rand(N,2)
sourcesC22 = np.random.rand(N,2)
sourcesC23 = np.random.rand(N,2)
sourcesC24 = np.random.rand(N,2)
sourcesC25 = np.random.rand(N,2)
sourcesC26 = np.random.rand(N,2)
sourcesC27 = np.random.rand(N,2)

for i in range(0,(2*N)):
  sourcesP1[i,0] = sourcesP1[i,0]*(P1d1-P1c1)+P1c1
  sourcesP1[i,1] = sourcesP1[i,1]*(P1d2-P1c2)+P1c2
for i in range(0,(2*N)):
  sourcesP2[i,0] = sourcesP2[i,0]*(P2d1-P2c1)+P2c1
  sourcesP2[i,1] = sourcesP2[i,1]*(P2d2-P2c2)+P2c2
for i in range(0,(2*N)):
  sourcesP3[i,0] = sourcesP3[i,0]*(P3d1-P3c1)+P3c1
  sourcesP3[i,1] = sourcesP3[i,1]*(P3d2-P3c2)+P3c2
for i in range(0,(2*N)):
  sourcesP4[i,0] = sourcesP4[i,0]*(P4d1-P4c1)+P4c1
  sourcesP4[i,1] = sourcesP4[i,1]*(P4d2-P4c2)+P4c2
for i in range(0,(2*N)):
  sourcesP5[i,0] = sourcesP5[i,0]*(P5d1-P5c1)+P5c1
  sourcesP5[i,1] = sourcesP5[i,1]*(P5d2-P5c2)+P5c2
for i in range(0,(2*N)):
  sourcesP6[i,0] = sourcesP6[i,0]*(P6d1-P6c1)+P6c1
  sourcesP6[i,1] = sourcesP6[i,1]*(P6d2-P6c2)+P6c2
for i in range(0,(2*N)):
  sourcesP7[i,0] = sourcesP7[i,0]*(P7d1-P7c1)+P7c1
  sourcesP7[i,1] = sourcesP7[i,1]*(P7d2-P7c2)+P7c2

for i in range(0,N):
  sourcesC1[i,0] = sourcesC1[i,0]*(C1b1-C1a1)+C1a1
  sourcesC1[i,1] = sourcesC1[i,1]*(C1b2-C1a2)+C1a2
for i in range(0,N):
  sourcesC2[i,0] = sourcesC2[i,0]*(C2b1-C2a1)+C2a1
  sourcesC2[i,1] = sourcesC2[i,1]*(C2b2-C2a2)+C2a2
for i in range(0,N):
  sourcesC3[i,0] = sourcesC3[i,0]*(C3b1-C3a1)+C3a1
  sourcesC3[i,1] = sourcesC3[i,1]*(C3b2-C3a2)+C3a2
for i in range(0,N):
  sourcesC4[i,0] = sourcesC4[i,0]*(C4b1-C4a1)+C4a1
  sourcesC4[i,1] = sourcesC4[i,1]*(C4b2-C4a2)+C4a2
for i in range(0,N):
  sourcesC5[i,0] = sourcesC5[i,0]*(C5b1-C5a1)+C5a1
  sourcesC5[i,1] = sourcesC5[i,1]*(C5b2-C5a2)+C5a2
for i in range(0,N):
  sourcesC6[i,0] = sourcesC6[i,0]*(C6b1-C6a1)+C6a1
  sourcesC6[i,1] = sourcesC6[i,1]*(C6b2-C6a2)+C6a2
for i in range(0,N):
  sourcesC7[i,0] = sourcesC7[i,0]*(C7b1-C7a1)+C7a1
  sourcesC7[i,1] = sourcesC7[i,1]*(C7b2-C7a2)+C7a2
for i in range(0,N):
  sourcesC8[i,0] = sourcesC8[i,0]*(C8b1-C8a1)+C8a1
  sourcesC8[i,1] = sourcesC8[i,1]*(C8b2-C8a2)+C8a2
for i in range(0,N):
  sourcesC9[i,0] = sourcesC9[i,0]*(C9b1-C9a1)+C9a1
  sourcesC9[i,1] = sourcesC9[i,1]*(C9b2-C9a2)+C9a2
for i in range(0,N):
  sourcesC10[i,0] = sourcesC10[i,0]*(C10b1-C10a1)+C10a1
  sourcesC10[i,1] = sourcesC10[i,1]*(C10b2-C10a2)+C10a2
for i in range(0,N):
  sourcesC11[i,0] = sourcesC11[i,0]*(C11b1-C11a1)+C11a1
  sourcesC11[i,1] = sourcesC11[i,1]*(C11b2-C11a2)+C11a2
for i in range(0,N):
  sourcesC12[i,0] = sourcesC12[i,0]*(C12b1-C12a1)+C12a1
  sourcesC12[i,1] = sourcesC12[i,1]*(C12b2-C12a2)+C12a2
for i in range(0,N):
  sourcesC13[i,0] = sourcesC13[i,0]*(C13b1-C13a1)+C13a1
  sourcesC13[i,1] = sourcesC13[i,1]*(C13b2-C13a2)+C13a2
for i in range(0,N):
  sourcesC14[i,0] = sourcesC14[i,0]*(C14b1-C14a1)+C14a1
  sourcesC14[i,1] = sourcesC14[i,1]*(C14b2-C14a2)+C14a2
for i in range(0,N):
  sourcesC15[i,0] = sourcesC15[i,0]*(C15b1-C15a1)+C15a1
  sourcesC15[i,1] = sourcesC15[i,1]*(C15b2-C15a2)+C15a2
for i in range(0,N):
  sourcesC16[i,0] = sourcesC16[i,0]*(C16b1-C16a1)+C16a1
  sourcesC16[i,1] = sourcesC16[i,1]*(C16b2-C16a2)+C16a2
for i in range(0,N):
  sourcesC17[i,0] = sourcesC17[i,0]*(C17b1-C17a1)+C17a1
  sourcesC17[i,1] = sourcesC17[i,1]*(C17b2-C17a2)+C17a2
for i in range(0,N):
  sourcesC18[i,0] = sourcesC18[i,0]*(C18b1-C18a1)+C18a1
  sourcesC18[i,1] = sourcesC18[i,1]*(C18b2-C18a2)+C18a2
for i in range(0,N):
  sourcesC19[i,0] = sourcesC19[i,0]*(C19b1-C19a1)+C19a1
  sourcesC19[i,1] = sourcesC19[i,1]*(C19b2-C19a2)+C19a2
for i in range(0,N):
  sourcesC20[i,0] = sourcesC20[i,0]*(C20b1-C20a1)+C20a1
  sourcesC20[i,1] = sourcesC20[i,1]*(C20b2-C20a2)+C20a2
for i in range(0,N):
  sourcesC21[i,0] = sourcesC21[i,0]*(C21b1-C21a1)+C21a1
  sourcesC21[i,1] = sourcesC21[i,1]*(C21b2-C21a2)+C21a2
for i in range(0,N):
  sourcesC22[i,0] = sourcesC22[i,0]*(C22b1-C22a1)+C22a1
  sourcesC22[i,1] = sourcesC22[i,1]*(C22b2-C22a2)+C22a2
for i in range(0,N):
  sourcesC23[i,0] = sourcesC23[i,0]*(C23b1-C23a1)+C23a1
  sourcesC23[i,1] = sourcesC23[i,1]*(C23b2-C23a2)+C23a2
for i in range(0,N):
  sourcesC24[i,0] = sourcesC24[i,0]*(C24b1-C24a1)+C24a1
  sourcesC24[i,1] = sourcesC24[i,1]*(C24b2-C24a2)+C24a2
for i in range(0,N):
  sourcesC25[i,0] = sourcesC25[i,0]*(C25b1-C25a1)+C25a1
  sourcesC25[i,1] = sourcesC25[i,1]*(C25b2-C25a2)+C25a2
for i in range(0,N):
  sourcesC26[i,0] = sourcesC26[i,0]*(C26b1-C26a1)+C26a1
  sourcesC26[i,1] = sourcesC26[i,1]*(C26b2-C26a2)+C26a2
for i in range(0,N):
  sourcesC27[i,0] = sourcesC27[i,0]*(C27b1-C27a1)+C27a1
  sourcesC27[i,1] = sourcesC27[i,1]*(C27b2-C27a2)+C27a2

# create target
point = [3.1,3.3]


for i in range(0,(2*N)):
  plt.scatter(sourcesP1[i,0],sourcesP1[i,1],color='black')
  plt.scatter(sourcesP2[i,0],sourcesP2[i,1],color='red')
  plt.scatter(sourcesP3[i,0],sourcesP3[i,1],color='blue')
  plt.scatter(sourcesP4[i,0],sourcesP4[i,1],color='yellow')
  plt.scatter(sourcesP5[i,0],sourcesP5[i,1],color='black')
  plt.scatter(sourcesP6[i,0],sourcesP6[i,1],color='red')
  plt.scatter(sourcesP7[i,0],sourcesP7[i,1],color='blue')
for i in range(0,N):
  plt.scatter(sourcesC1[i,0],sourcesC1[i,1],color='yellow')
  plt.scatter(sourcesC2[i,0],sourcesC2[i,1],color='black')
  plt.scatter(sourcesC3[i,0],sourcesC3[i,1],color='red')
  plt.scatter(sourcesC4[i,0],sourcesC4[i,1],color='blue')
  plt.scatter(sourcesC5[i,0],sourcesC5[i,1],color='yellow')
  plt.scatter(sourcesC6[i,0],sourcesC6[i,1],color='black')
  plt.scatter(sourcesC7[i,0],sourcesC7[i,1],color='red')
  plt.scatter(sourcesC8[i,0],sourcesC8[i,1],color='blue')
  plt.scatter(sourcesC9[i,0],sourcesC9[i,1],color='yellow')
  plt.scatter(sourcesC10[i,0],sourcesC10[i,1],color='black')
  plt.scatter(sourcesC11[i,0],sourcesC11[i,1],color='red')
  plt.scatter(sourcesC12[i,0],sourcesC12[i,1],color='blue')
  plt.scatter(sourcesC13[i,0],sourcesC13[i,1],color='yellow')
  plt.scatter(sourcesC14[i,0],sourcesC14[i,1],color='black')
  plt.scatter(sourcesC15[i,0],sourcesC15[i,1],color='red')
  plt.scatter(sourcesC16[i,0],sourcesC16[i,1],color='blue')
  plt.scatter(sourcesC17[i,0],sourcesC17[i,1],color='yellow')
  plt.scatter(sourcesC18[i,0],sourcesC18[i,1],color='black')
  plt.scatter(sourcesC19[i,0],sourcesC19[i,1],color='red')
  plt.scatter(sourcesC20[i,0],sourcesC20[i,1],color='blue')
  plt.scatter(sourcesC21[i,0],sourcesC21[i,1],color='yellow')
  plt.scatter(sourcesC22[i,0],sourcesC22[i,1],color='black')
  plt.scatter(sourcesC23[i,0],sourcesC23[i,1],color='red')
  plt.scatter(sourcesC24[i,0],sourcesC24[i,1],color='blue')
  plt.scatter(sourcesC25[i,0],sourcesC25[i,1],color='yellow')
  plt.scatter(sourcesC26[i,0],sourcesC26[i,1],color='black')
  plt.scatter(sourcesC27[i,0],sourcesC27[i,1],color='red')
  plt.scatter(point[0],point[1],color='green')
plt.grid()
plt.show()


# create some charge (density) for each source
# they can be +1 or -1
# notice how we handle each child box's source charges separately
sigmaP1 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP1[i] = (np.random.randint(0,2)*2)-1

sigmaP2 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP2[i] = (np.random.randint(0,2)*2)-1
  
sigmaP3 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP3[i] = (np.random.randint(0,2)*2)-1

sigmaP4 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP4[i] = (np.random.randint(0,2)*2)-1

sigmaP5 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP5[i] = (np.random.randint(0,2)*2)-1

sigmaP6 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP6[i] = (np.random.randint(0,2)*2)-1

sigmaP7 = np.zeros(shape=((2*N),1))
for i in range(0,(2*N)):
  sigmaP7[i] = (np.random.randint(0,2)*2)-1


sigmaC1 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC1[i] = (np.random.randint(0,2)*2)-1

sigmaC2 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC2[i] = (np.random.randint(0,2)*2)-1

sigmaC3 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC3[i] = (np.random.randint(0,2)*2)-1

sigmaC4 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC4[i] = (np.random.randint(0,2)*2)-1

sigmaC5 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC5[i] = (np.random.randint(0,2)*2)-1

sigmaC6 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC6[i] = (np.random.randint(0,2)*2)-1

sigmaC7 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC7[i] = (np.random.randint(0,2)*2)-1

sigmaC8 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC8[i] = (np.random.randint(0,2)*2)-1

sigmaC9 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC9[i] = (np.random.randint(0,2)*2)-1

sigmaC10 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC10[i] = (np.random.randint(0,2)*2)-1

sigmaC11 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC11[i] = (np.random.randint(0,2)*2)-1

sigmaC12 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC12[i] = (np.random.randint(0,2)*2)-1

sigmaC13 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC13[i] = (np.random.randint(0,2)*2)-1

sigmaC14 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC14[i] = (np.random.randint(0,2)*2)-1

sigmaC15 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC15[i] = (np.random.randint(0,2)*2)-1

sigmaC16 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC16[i] = (np.random.randint(0,2)*2)-1

sigmaC17 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC17[i] = (np.random.randint(0,2)*2)-1

sigmaC18 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC18[i] = (np.random.randint(0,2)*2)-1

sigmaC19 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC19[i] = (np.random.randint(0,2)*2)-1

sigmaC20 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC20[i] = (np.random.randint(0,2)*2)-1

sigmaC21 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC21[i] = (np.random.randint(0,2)*2)-1

sigmaC22 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC22[i] = (np.random.randint(0,2)*2)-1

sigmaC23 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC23[i] = (np.random.randint(0,2)*2)-1

sigmaC24 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC24[i] = (np.random.randint(0,2)*2)-1

sigmaC25 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC25[i] = (np.random.randint(0,2)*2)-1

sigmaC26 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC26[i] = (np.random.randint(0,2)*2)-1

sigmaC27 = np.zeros(shape=(N,1))
for i in range(0,N):
  sigmaC27[i] = (np.random.randint(0,2)*2)-1

''' these are the weights for each interaction list interaction '''

# each Chebyshev node, m, gets a weight
def WP1(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P1c1,P1d1)[m1],nodes(n,P1c2,P1d2)[m2],sourcesP1[j,0],sourcesP1[j,1],P1c1,P1d1,P1c2,P1d2)*sigmaP1[j]
  return sum

def WP2(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P2c1,P2d1)[m1],nodes(n,P2c2,P2d2)[m2],sourcesP2[j,0],sourcesP2[j,1],P2c1,P2d1,P2c2,P2d2)*sigmaP2[j]
  return sum

def WP3(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P3c1,P3d1)[m1],nodes(n,P3c2,P3d2)[m2],sourcesP3[j,0],sourcesP3[j,1],P3c1,P3d1,P3c2,P3d2)*sigmaP3[j]
  return sum

def WP4(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P4c1,P4d1)[m1],nodes(n,P4c2,P4d2)[m2],sourcesP4[j,0],sourcesP4[j,1],P4c1,P4d1,P4c2,P4d2)*sigmaP4[j]
  return sum

def WP5(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P5c1,P5d1)[m1],nodes(n,P5c2,P5d2)[m2],sourcesP5[j,0],sourcesP5[j,1],P5c1,P5d1,P5c2,P5d2)*sigmaP5[j]
  return sum

def WP6(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P6c1,P6d1)[m1],nodes(n,P6c2,P6d2)[m2],sourcesP6[j,0],sourcesP6[j,1],P6c1,P6d1,P6c2,P6d2)*sigmaP6[j]
  return sum

def WP7(m1,m2):
  sum = 0
  for j in range(0,(2*N)):
    sum += R(n,nodes(n,P7c1,P7d1)[m1],nodes(n,P7c2,P7d2)[m2],sourcesP7[j,0],sourcesP7[j,1],P7c1,P7d1,P7c2,P7d2)*sigmaP7[j]
  return sum


def WC1(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C1a1,C1b1)[m1],nodes(n,C1a2,C1b2)[m2],sourcesC1[j,0],sourcesC1[j,1],C1a1,C1b1,C1a2,C1b2)*sigmaC1[j]
  return sum

def WC2(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C2a1,C2b1)[m1],nodes(n,C2a2,C2b2)[m2],sourcesC2[j,0],sourcesC2[j,1],C2a1,C2b1,C2a2,C2b2)*sigmaC2[j]
  return sum

def WC3(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C3a1,C3b1)[m1],nodes(n,C3a2,C3b2)[m2],sourcesC3[j,0],sourcesC3[j,1],C3a1,C3b1,C3a2,C3b2)*sigmaC3[j]
  return sum

def WC4(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C4a1,C4b1)[m1],nodes(n,C4a2,C4b2)[m2],sourcesC4[j,0],sourcesC4[j,1],C4a1,C4b1,C4a2,C4b2)*sigmaC4[j]
  return sum

def WC5(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C5a1,C5b1)[m1],nodes(n,C5a2,C5b2)[m2],sourcesC5[j,0],sourcesC5[j,1],C5a1,C5b1,C5a2,C5b2)*sigmaC5[j]
  return sum

def WC6(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C6a1,C6b1)[m1],nodes(n,C6a2,C6b2)[m2],sourcesC6[j,0],sourcesC6[j,1],C6a1,C6b1,C6a2,C6b2)*sigmaC6[j]
  return sum

def WC7(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C7a1,C7b1)[m1],nodes(n,C7a2,C7b2)[m2],sourcesC7[j,0],sourcesC7[j,1],C7a1,C7b1,C7a2,C7b2)*sigmaC7[j]
  return sum

def WC8(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C8a1,C8b1)[m1],nodes(n,C8a2,C8b2)[m2],sourcesC8[j,0],sourcesC8[j,1],C8a1,C8b1,C8a2,C8b2)*sigmaC8[j]
  return sum

def WC9(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C9a1,C9b1)[m1],nodes(n,C9a2,C9b2)[m2],sourcesC9[j,0],sourcesC9[j,1],C9a1,C9b1,C9a2,C9b2)*sigmaC9[j]
  return sum

def WC10(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C10a1,C10b1)[m1],nodes(n,C10a2,C10b2)[m2],sourcesC10[j,0],sourcesC10[j,1],C10a1,C10b1,C10a2,C10b2)*sigmaC10[j]
  return sum

def WC11(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C11a1,C11b1)[m1],nodes(n,C11a2,C11b2)[m2],sourcesC11[j,0],sourcesC11[j,1],C11a1,C11b1,C11a2,C11b2)*sigmaC11[j]
  return sum

def WC12(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C12a1,C12b1)[m1],nodes(n,C12a2,C12b2)[m2],sourcesC12[j,0],sourcesC12[j,1],C12a1,C12b1,C12a2,C12b2)*sigmaC12[j]
  return sum

def WC13(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C13a1,C13b1)[m1],nodes(n,C13a2,C13b2)[m2],sourcesC13[j,0],sourcesC13[j,1],C13a1,C13b1,C13a2,C13b2)*sigmaC13[j]
  return sum

def WC14(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C14a1,C14b1)[m1],nodes(n,C14a2,C14b2)[m2],sourcesC14[j,0],sourcesC14[j,1],C14a1,C14b1,C14a2,C14b2)*sigmaC14[j]
  return sum

def WC15(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C15a1,C15b1)[m1],nodes(n,C15a2,C15b2)[m2],sourcesC15[j,0],sourcesC15[j,1],C15a1,C15b1,C15a2,C15b2)*sigmaC15[j]
  return sum

def WC16(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C16a1,C16b1)[m1],nodes(n,C16a2,C16b2)[m2],sourcesC16[j,0],sourcesC16[j,1],C16a1,C16b1,C16a2,C16b2)*sigmaC16[j]
  return sum

def WC17(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C17a1,C17b1)[m1],nodes(n,C17a2,C17b2)[m2],sourcesC17[j,0],sourcesC17[j,1],C17a1,C17b1,C17a2,C17b2)*sigmaC17[j]
  return sum

def WC18(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C18a1,C18b1)[m1],nodes(n,C18a2,C18b2)[m2],sourcesC18[j,0],sourcesC18[j,1],C18a1,C18b1,C18a2,C18b2)*sigmaC18[j]
  return sum

def WC19(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C19a1,C19b1)[m1],nodes(n,C19a2,C19b2)[m2],sourcesC19[j,0],sourcesC19[j,1],C19a1,C19b1,C19a2,C19b2)*sigmaC19[j]
  return sum

def WC20(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C20a1,C20b1)[m1],nodes(n,C20a2,C20b2)[m2],sourcesC20[j,0],sourcesC20[j,1],C20a1,C20b1,C20a2,C20b2)*sigmaC20[j]
  return sum

def WC21(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C21a1,C21b1)[m1],nodes(n,C21a2,C21b2)[m2],sourcesC21[j,0],sourcesC21[j,1],C21a1,C21b1,C21a2,C21b2)*sigmaC21[j]
  return sum

def WC22(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C22a1,C22b1)[m1],nodes(n,C22a2,C22b2)[m2],sourcesC22[j,0],sourcesC22[j,1],C22a1,C22b1,C22a2,C22b2)*sigmaC22[j]
  return sum

def WC23(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C23a1,C23b1)[m1],nodes(n,C23a2,C23b2)[m2],sourcesC23[j,0],sourcesC23[j,1],C23a1,C23b1,C23a2,C23b2)*sigmaC23[j]
  return sum

def WC24(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C24a1,C24b1)[m1],nodes(n,C24a2,C24b2)[m2],sourcesC24[j,0],sourcesC24[j,1],C24a1,C24b1,C24a2,C24b2)*sigmaC24[j]
  return sum

def WC25(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C25a1,C25b1)[m1],nodes(n,C25a2,C25b2)[m2],sourcesC25[j,0],sourcesC25[j,1],C25a1,C25b1,C25a2,C25b2)*sigmaC25[j]
  return sum

def WC26(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C26a1,C26b1)[m1],nodes(n,C26a2,C26b2)[m2],sourcesC26[j,0],sourcesC26[j,1],C26a1,C26b1,C26a2,C26b2)*sigmaC26[j]
  return sum

def WC27(m1,m2):
  sum = 0
  for j in range(0,N):
    sum += R(n,nodes(n,C27a1,C27b1)[m1],nodes(n,C27a2,C27b2)[m2],sourcesC27[j,0],sourcesC27[j,1],C27a1,C27b1,C27a2,C27b2)*sigmaC27[j]
  return sum

''' compute multipole to local estimate '''
def gP(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P1c1,P1d1)[m1],nodes(n,P1c2,P1d2)[m2])*WP1(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P2c1,P2d1)[m1],nodes(n,P2c2,P2d2)[m2])*WP2(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P3c1,P3d1)[m1],nodes(n,P3c2,P3d2)[m2])*WP3(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P4c1,P4d1)[m1],nodes(n,P4c2,P4d2)[m2])*WP4(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P5c1,P5d1)[m1],nodes(n,P5c2,P5d2)[m2])*WP5(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P6c1,P6d1)[m1],nodes(n,P6c2,P6d2)[m2])*WP6(m1,m2) + log(nodes(n,Pa1,Pb1)[l1],nodes(n,Pa2,Pb2)[l2],nodes(n,P7c1,P7d1)[m1],nodes(n,P7c2,P7d2)[m2])*WP7(m1,m2)
  return sum

def gC(l1,l2):
  sum = 0
  for m1 in range(0,n):
    for m2 in range(0,n):
      sum += log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C1a1,C1b1)[m1],nodes(n,C1a2,C1b2)[m2])*WC1(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C2a1,C2b1)[m1],nodes(n,C2a2,C2b2)[m2])*WC2(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C3a1,C3b1)[m1],nodes(n,C3a2,C3b2)[m2])*WC3(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C4a1,C4b1)[m1],nodes(n,C4a2,C4b2)[m2])*WC4(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C5a1,C5b1)[m1],nodes(n,C5a2,C5b2)[m2])*WC5(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C6a1,C6b1)[m1],nodes(n,C6a2,C6b2)[m2])*WC6(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C7a1,C7b1)[m1],nodes(n,C7a2,C7b2)[m2])*WC7(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C8a1,C8b1)[m1],nodes(n,C8a2,C8b2)[m2])*WC8(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C9a1,C9b1)[m1],nodes(n,C9a2,C9b2)[m2])*WC9(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C10a1,C10b1)[m1],nodes(n,C10a2,C10b2)[m2])*WC10(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C11a1,C11b1)[m1],nodes(n,C11a2,C11b2)[m2])*WC11(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C12a1,C12b1)[m1],nodes(n,C12a2,C12b2)[m2])*WC12(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C13a1,C13b1)[m1],nodes(n,C13a2,C13b2)[m2])*WC13(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C14a1,C14b1)[m1],nodes(n,C14a2,C14b2)[m2])*WC14(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C15a1,C15b1)[m1],nodes(n,C15a2,C15b2)[m2])*WC15(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C16a1,C16b1)[m1],nodes(n,C16a2,C16b2)[m2])*WC16(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C17a1,C17b1)[m1],nodes(n,C17a2,C17b2)[m2])*WC17(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C18a1,C18b1)[m1],nodes(n,C18a2,C18b2)[m2])*WC18(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C19a1,C19b1)[m1],nodes(n,C19a2,C19b2)[m2])*WC19(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C20a1,C20b1)[m1],nodes(n,C20a2,C20b2)[m2])*WC20(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C21a1,C21b1)[m1],nodes(n,C21a2,C21b2)[m2])*WC21(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C22a1,C22b1)[m1],nodes(n,C22a2,C22b2)[m2])*WC22(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C23a1,C23b1)[m1],nodes(n,C23a2,C23b2)[m2])*WC23(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C24a1,C24b1)[m1],nodes(n,C24a2,C24b2)[m2])*WC24(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C25a1,C25b1)[m1],nodes(n,C25a2,C25b2)[m2])*WC25(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C26a1,C26b1)[m1],nodes(n,C26a2,C26b2)[m2])*WC26(m1,m2) + log(nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,C27a1,C27b1)[m1],nodes(n,C27a2,C27b2)[m2])*WC27(m1,m2)
  return sum

''' compute estimate for local expansion '''
''' note there is no local to local contribution b/c parent has no interaction list '''
def f(l1,l2):
  sum = gC(l1,l2)
  for lprime1 in range(0,n):
    for lprime2 in range(0,n):
      sum += gP(lprime1,lprime2)*R(n,nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],nodes(n,Pa1,Pb1)[lprime1],nodes(n,Pa2,Pb2)[lprime2],Pa1,Pb1,Pa2,Pb2)
  return sum

fest = 0
for l1 in range(0,n):
  for l2 in range(0,n):
    fest += f(l1,l2)*R(n,nodes(n,a1,b1)[l1],nodes(n,a2,b2)[l2],point[0],point[1],a1,b1,a2,b2)

print("Estimated potential:")
print(fest)

''' computation of actual local expansion '''
fact1 = 0
for j in range(0,(2*N)):
  fact1 += log(point[0],point[1],sourcesP1[j,0],sourcesP1[j,1])*sigmaP1[j] + log(point[0],point[1],sourcesP2[j,0],sourcesP2[j,1])*sigmaP2[j] + log(point[0],point[1],sourcesP3[j,0],sourcesP3[j,1])*sigmaP3[j] + log(point[0],point[1],sourcesP4[j,0],sourcesP4[j,1])*sigmaP4[j] + log(point[0],point[1],sourcesP5[j,0],sourcesP5[j,1])*sigmaP5[j] + log(point[0],point[1],sourcesP6[j,0],sourcesP6[j,1])*sigmaP6[j] + log(point[0],point[1],sourcesP7[j,0],sourcesP7[j,1])*sigmaP7[j]

fact2 = 0
for j in range(0,N):
  fact2 += log(point[0],point[1],sourcesC1[j,0],sourcesC1[j,1])*sigmaC1[j] + log(point[0],point[1],sourcesC2[j,0],sourcesC2[j,1])*sigmaC2[j] + log(point[0],point[1],sourcesC3[j,0],sourcesC3[j,1])*sigmaC3[j] + log(point[0],point[1],sourcesC4[j,0],sourcesC4[j,1])*sigmaC4[j] + log(point[0],point[1],sourcesC5[j,0],sourcesC5[j,1])*sigmaC5[j] + log(point[0],point[1],sourcesC6[j,0],sourcesC6[j,1])*sigmaC6[j] + log(point[0],point[1],sourcesC7[j,0],sourcesC7[j,1])*sigmaC7[j] + log(point[0],point[1],sourcesC8[j,0],sourcesC8[j,1])*sigmaC8[j] + log(point[0],point[1],sourcesC9[j,0],sourcesC9[j,1])*sigmaC9[j] + log(point[0],point[1],sourcesC10[j,0],sourcesC10[j,1])*sigmaC10[j] + log(point[0],point[1],sourcesC11[j,0],sourcesC11[j,1])*sigmaC11[j] + log(point[0],point[1],sourcesC12[j,0],sourcesC12[j,1])*sigmaC12[j] + log(point[0],point[1],sourcesC13[j,0],sourcesC13[j,1])*sigmaC13[j] + log(point[0],point[1],sourcesC14[j,0],sourcesC14[j,1])*sigmaC14[j] + log(point[0],point[1],sourcesC15[j,0],sourcesC15[j,1])*sigmaC15[j] + log(point[0],point[1],sourcesC16[j,0],sourcesC16[j,1])*sigmaC16[j] + log(point[0],point[1],sourcesC17[j,0],sourcesC17[j,1])*sigmaC17[j] + log(point[0],point[1],sourcesC18[j,0],sourcesC18[j,1])*sigmaC18[j] + log(point[0],point[1],sourcesC19[j,0],sourcesC19[j,1])*sigmaC19[j] + log(point[0],point[1],sourcesC20[j,0],sourcesC20[j,1])*sigmaC20[j] + log(point[0],point[1],sourcesC21[j,0],sourcesC21[j,1])*sigmaC21[j] + log(point[0],point[1],sourcesC22[j,0],sourcesC22[j,1])*sigmaC22[j] + log(point[0],point[1],sourcesC23[j,0],sourcesC23[j,1])*sigmaC23[j] + log(point[0],point[1],sourcesC24[j,0],sourcesC24[j,1])*sigmaC24[j] + log(point[0],point[1],sourcesC25[j,0],sourcesC25[j,1])*sigmaC25[j] + log(point[0],point[1],sourcesC26[j,0],sourcesC26[j,1])*sigmaC26[j] + log(point[0],point[1],sourcesC27[j,0],sourcesC27[j,1])*sigmaC27[j]

fact = fact1 + fact2

print("Actual potential:")
print(fact)