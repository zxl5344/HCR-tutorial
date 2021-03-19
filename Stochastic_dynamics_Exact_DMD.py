import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt


def Standard_DMD(X,Y):
	try:
		#find SVD
		U,s,Vh = LA.svd(X, full_matrices=False)      		#Vh is the complex conjugate; full_matrices is disabled

		V = np.conjugate(Vh)
		V = V.T					 
		Uh = np.conjugate(U)					
		Uh = Uh.T						#Uh is the complex conjugate of U

		s = np.diag(s)			 			#resize the s		
		s_inv = LA.inv(s)					#s_inv is the inverse of s

		#define Nex matrix
		A_Tilde = Uh@Y@V@s_inv

		#eig of A_Tilde 
		lamda,omega = LA.eig(A_Tilde)

		#define phi
		for i in range(0,len(lamda)): 
			omega[:,i] *=lamda[i]

		phi = Y@V@LA.inv(np.diag(lamda))@omega
		return lamda,phi
	except:
		U = 1; s = LA.norm(X); Vh = X/s				

		s_inv = 1/s
		V = Vh.T
		
		A_Tilde = U*Y@V*s_inv
		return A_Tilde


lamda = 0.5
n = np.random.normal(0, np.sqrt(5),999)
z = np.zeros(1000)
k = np.zeros(1000)

for i in range(0,999):
	k[i+1] = i+1
	z[i+1] = lamda*z[i] + n[i]


X = z[0:999]
Y = z[1:1000]


A = Standard_DMD(X,Y) 
print(A)

#plot z_k with k
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(k,z)

ax1.set(xlabel='k', ylabel='z_k')
ax1.grid()

#plot z_k+1 with z_k

ax2.scatter(X,Y)
x = np.linspace(-10,10,100)
y_DMD = A*x
y_true = lamda*x
ax2.plot(x, y_DMD, color='red', label = 'DMD fit')
ax2.plot(x, y_true, color='black', label = 'true slope')

ax2.set(xlabel='z_k', ylabel='z_k+1')
ax2.grid()
ax2.legend(shadow=True, fancybox=True)

plt.show()