import numpy as np
from numpy import linalg as LA
import pandas as pd

def Standard_DMD(X,Y):
	#find SVD
	U = 1; s = 1; Vh = X     		#Vh is the complex conjugate; full_matrices is disabled

	V = np.conjugate(Vh)
	V = LA.inv(Vh)				 						
	
	s_inv = 1/s					#s_inv is the inverse of s

	#define Nex matrix
	A_Tilde = U*Y@V*s_inv
	print(A_Tilde)
	return A_Tilde


lamda = 0.5
n = np.random.normal(0, np.sqrt(5),999)
z = np.zeros(1000)

for i in range(0,999):
	z[i+1] = lamda*z[i] + n[i]


X = z[0:999]
Y = z[1:1000]


Standard_DMD(X,Y) 