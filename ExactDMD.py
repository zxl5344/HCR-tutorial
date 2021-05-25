import numpy as np
from numpy import linalg as LA
import pandas as pd
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import style
import hdbscan
import seaborn as sns

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

def Koopman(X, Y):
    for i in range(0, len(X[0,:])):
        if i == 0:
            phi = X[:, i].reshape(10, 1)
            phi1 = Y[:, i].reshape(10,1)
            G = phi @ phi.T
            A = phi @ phi1.T
        else:
            phi = X[:, i].reshape(10, 1)
            phi1 = Y[:, i].reshape(10,1)
            G += phi @ phi.T
            A += phi @ phi1.T
    print((G.shape))
    G = 1/len(X[0, :])*G
    A = 1/len(X[0, :])*A
    K = LA.pinv(G)*A

    return K


windowsize = 30
overlap = 5
# arrange the data

for i in range(0, 30):
    X = data.iloc[(1 + i * 1802):(1801 + i * 1802), 0:5]  # Take 1(z_0) to 1801(z_m-1) rows and 2 to 5 columns as X matrix
    Y = data.iloc[(2 + i * 1802):(1802 + i * 1802), 0:5]  # Take 2(z_0) to 1802(z_m) rows and 2 to 5 columns as Y matrix
    uX_cos = np.multiply(np.cos(X.iloc[:, 0]), X.iloc[:, 4])
    uY_cos = np.multiply(np.cos(Y.iloc[:, 0]), Y.iloc[:, 4])
    uXdot_cos = np.multiply(np.cos(X.iloc[:, 2]), X.iloc[:, 4])
    uYdot_cos = np.multiply(np.cos(Y.iloc[:, 2]), Y.iloc[:, 4])
    uXcos2 = np.multiply(np.cos(X.iloc[:, 4] * pi / 20), np.cos(X.iloc[:, 4] * pi / 20))
    uYcos2 = np.multiply(np.cos(Y.iloc[:, 4] * pi / 20), np.cos(Y.iloc[:, 4] * pi / 20))
    uXdot2 = np.multiply(X.iloc[:, 3], X.iloc[:, 3])
    uYdot2 = np.multiply(Y.iloc[:, 3], Y.iloc[:, 3])
    Xones = np.ones(1800)
    Yones = np.ones(1800)

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
    K = np.zeros((int(1800/windowsize), 100))
    for j in range(0, int(1800/windowsize)):
        if j == 0:
            A = Exact_DMD(X[:, 0:windowsize*(j+1)+overlap], Y[:, 0:windowsize*(j+1)+overlap])
        elif j == 1800/windowsize - 1:
            A = Exact_DMD(X[:, windowsize*j-overlap:windowsize*(j+1)], Y[:, windowsize*j-overlap:windowsize*(j+1)])
        else:
            A = Exact_DMD(X[:, windowsize*j-overlap:windowsize*(j+1)+overlap], Y[:, windowsize*j-overlap:windowsize*(j+1)+overlap])
        K[j, :] = A.flatten()

    if i == 0:
        K1 = K
    else:
        K1 = np.append(K1, K, axis=0)

print(K1.shape)

#clustering
print("Start clustering")
clusterer = hdbscan.HDBSCAN()
clusterer.fit(K1)
#printout the label
print(max(clusterer.labels_))

clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette("deep", 8))
plt.show()

"""
i = 29

X = data.iloc[(1 + i * 1802):(1801 + i * 1802), 0:5]  # Take 1(z_0) to 1801(z_m-1) rows and 2 to 5 columns as X matrix
Y = data.iloc[(2 + i * 1802):(1802 + i * 1802), 0:5]  # Take 2(z_0) to 1802(z_m) rows and 2 to 5 columns as Y matrix
uX_cos = np.multiply(np.cos(X.iloc[:, 0]), X.iloc[:, 4])
uY_cos = np.multiply(np.cos(Y.iloc[:, 0]), Y.iloc[:, 4])
uXdot_cos = np.multiply(np.cos(X.iloc[:, 2]), X.iloc[:, 4])
uYdot_cos = np.multiply(np.cos(Y.iloc[:, 2]), Y.iloc[:, 4])
uXcos2 = np.multiply(np.cos(X.iloc[:, 4] * pi / 20), np.cos(X.iloc[:, 4] * pi / 20))
uYcos2 = np.multiply(np.cos(Y.iloc[:, 4] * pi / 20), np.cos(Y.iloc[:, 4] * pi / 20))
uXdot2 = np.multiply(X.iloc[:, 3], X.iloc[:, 3])
uYdot2 = np.multiply(Y.iloc[:, 3], Y.iloc[:, 3])
Xones = np.ones(1800)
Yones = np.ones(1800)

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
K = Koopman(X1, Y1)

print("X1 ", X1.shape)
print("Y1 ", Y1.shape)


#simulation
errorq = np.zeros(1800)
errorx = np.zeros(1800)
errorqdot = np.zeros(1800)
errorxdot = np.zeros(1800)
X_test = X1[:, 0]
err_max = 0
time = 0

for k in range(0, 1800):
    Y_test = Y1[:, k]
    Y_est = LA.matrix_power(np.eye(10)+K*1/60, k) @ X_test
    Y_est = A*X1[:, k]
    errorq[k] = LA.norm(Y_test[1] - Y_est[1]) / LA.norm(Y_test[1])
    errorx[k] = LA.norm(Y_test[2] - Y_est[2]) / LA.norm(Y_test[2])
    if LA.norm(Y_est[3]) == 0:
        errorqdot[k] = 0
    else:
        errorqdot[k] = LA.norm(Y_test[3] - Y_est[3]) / LA.norm(Y_test[3])
    errorxdot[k] = LA.norm(Y_test[4] - Y_est[4]) / LA.norm(Y_test[4])




t = np.arange(0, 1800)
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
"""





