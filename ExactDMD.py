import numpy as np
from numpy import linalg as LA
import pandas as pd
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import style

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
"""for i in range(0, 30):
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
        Y1 = np.append(Y1, Y, axis=1)"""

i = 5

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

X1 = X.T  # X.T means transpose of X
Y1 = Y.T

A = Exact_DMD(X1, Y1)

print("X1 ", X1.shape)
print("Y1 ", Y1.shape)

pd.set_option('display.max_columns', None)
print("A\n", pd.DataFrame(A))

#simulation
errorq = np.zeros(1801)
errorx = np.zeros(1801)
errorqdot = np.zeros(1801)
errorxdot = np.zeros(1801)
X_test = X1[:, 0]
err_max = 0
time = 0

for k in range(0, 1801):
    Y_test = Y1[:, k]
    Y_est = LA.matrix_power(A, k) @ X_test
    errorq[k] = LA.norm(Y_test[1] - Y_est[1]) / LA.norm(Y_est[1])
    errorx[k] = LA.norm(Y_test[2] - Y_est[2]) / LA.norm(Y_est[2])
    if LA.norm(Y_est[3]) == 0:
        errorqdot[k] = 0
    else:
        errorqdot[k] = LA.norm(Y_test[3] - Y_est[3]) / LA.norm(Y_est[3])
    errorxdot[k] = LA.norm(Y_test[4] - Y_est[4]) / LA.norm(Y_est[4])
    """if err_max < errorq[k]:
        err_max = errorq[k]
        time = k


print("max error time", time)
print("max error", err_max)
"""

t = np.arange(0, 1801)
style.use("ggplot")
fig, axs = plt.subplots(2,2)
axs[0, 0].plot(t, errorq)
axs[0, 0].set_title('q')
axs[0, 1].plot(t, errorx)
axs[0, 1].set_title('x')
axs[1, 0].plot(t, errorqdot)
axs[1, 0].set_title('qdot')
axs[1, 1].plot(t, errorxdot)
axs[1, 1].set_title('xdot')
plt.show()






