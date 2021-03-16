from IPython.display import display
import numpy as np
from numpy import linalg as LA
from numpy import array 

#input matrix
A = np.array([[2, 0, 0],[0, 3 ,4], [0, 4, 9]]) 

#calculate the SVD
U,s,Vh = LA.svd(A)

#displays the result from SVD
display(U,s,Vh)

#verify the result
U@np.diag(s)@Vh