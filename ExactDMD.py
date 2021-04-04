import numpy as np
from numpy import linalg as LA
import pandas as pd
from numpy import pi

# read data from excel
data = pd.read_csv("#P_car_data.csv", sep=",")

data = data[["field.q0", "field.q1", "field.dq0", "field.dq1", "field.u0"]]


def Exact_DMD(X, Y):
    # find SVD
    U, s, Vh = LA.svd(X, full_matrices=False)  # Vh is the complex conjugate; full_matrices is disabled

    V = np.conjugate(Vh)
    V = V.T
    Uh = np.conjugate(U)
    Uh = Uh.T  # Uh is the complex conjugate of U

    s = np.diag(s)  # resize the s
    s_inv = LA.inv(s)  # s_inv is the inverse of s

    # define Nex matrix
    A_Tilde = Uh @ Y @ V @ s_inv

    # eig of A_Tilde
    lamda, omega = LA.eig(A_Tilde)

    # define phi

    phi = Y @ V @ s_inv @ omega @ LA.inv(np.diag(lamda))

    A = phi @ np.diag(lamda) @ LA.inv(phi)

    return A


# arrange the data
for i in range(0, 30):
    X = data.iloc[(0 + i * 1802):(1801 + i * 1802), 0:5]  # Take 1(z_0) to 1801(z_m-1) rows and 2 to 5 columns as X matrix
    Y = data.iloc[(1 + i * 1802):(1802 + i * 1802), 0:5]  # Take 2(z_0) to 1802(z_m) rows and 2 to 5 columns as Y matrix
    uX_cos = np.multiply(np.cos(X.iloc[:, 0]), X.iloc[:, 4])
    uY_cos = np.multiply(np.cos(Y.iloc[:, 0]), Y.iloc[:, 4])
    uXdot_cos = np.multiply(np.cos(X.iloc[:, 2]), X.iloc[:, 4])
    uYdot_cos = np.multiply(np.cos(Y.iloc[:, 2]), Y.iloc[:, 4])
    uXcos2 = np.multiply(np.cos(X.iloc[:, 4] * pi / 20), np.cos(X.iloc[:, 4] * pi / 20))
    uYcos2 = np.multiply(np.cos(Y.iloc[:, 4] * pi / 20), np.cos(Y.iloc[:, 4] * pi / 20))
    uXdot2 = np.multiply(X.iloc[:, 3], X.iloc[:, 3])
    uYdot2 = np.multiply(Y.iloc[:, 3], Y.iloc[:, 3])
    Xones = np.ones(1801)
    Yones = np.ones(1801)

    X = np.column_stack((X, uX_cos))
    Y = np.column_stack((Y, uY_cos))
    X = np.column_stack((X, uXdot_cos))
    Y = np.column_stack((Y, uYdot_cos))
    X = np.column_stack((X, uXcos2))
    Y = np.column_stack((Y, uYcos2))
    X = np.column_stack((X, uXdot2))
    Y = np.column_stack((Y, uYdot2))
    X = np.column_stack((X, Xones.T))
    Y = np.column_stack((Y, Yones.T))

    X = X.T  # X.T means transpose of X
    Y = Y.T

    if i == 0:
        X1 = X
        Y1 = Y
    else:
        X1 = np.append(X1, X, axis=1)
        Y1 = np.append(Y1, Y, axis=1)

A = Exact_DMD(X1, Y1)

"""print("X1 ", X1[:,1801])
print("Y1 ", Y1[:,1801])"""

"""pd.set_option('display.max_columns', None)
print("A\n", pd.DataFrame(A))"""

#simulation
test = 1000
X_init = X1[:, test]
Y_final = Y1[:, test]

Y_est = A @ X_init

print("Y final: ", Y_final)
print("Y_est: ", Y_est)




