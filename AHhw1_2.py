import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


# define ricatti equation
def riccati(P):
    return -P @ A - A.T @ P + P @ B / R @ B.T @ P - Q


def TBVBP(t, x, p):
    return np.vstack((A@x-B/R@B.T@p, -Q@x-A.T@p))


def bc(ya, yb, p):
    return np.array([X_initial, np.zeros((2, 1)), np.zeros((2, 1))])


# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rk4(sol_final, t, dt, f):
    # Iterate for number of iterations
    row = len(sol_final)
    columns = len(sol_final[0])
    sol = np.zeros((row, columns * len(t)))
    sol[:, (columns * len(t) - columns):(columns * len(t))] = sol_final
    for i in range(len(t) - 1, 0, -1):
        k1 = dt * f(sol[:, columns * i:(columns * i + columns)])
        k2 = dt * f(sol[:, columns * i:(columns * i + columns)] - k1 / 2.)
        k3 = dt * f(sol[:, columns * i:(columns * i + columns)] - k2 / 2.)
        k4 = dt * f(sol[:, columns * i:(columns * i + columns)] - k3)
        # Update next value of p
        sol[:, (columns * i - columns):columns * i] = sol[:, columns * i:(columns * i + columns)] - (1 / 6.) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return sol


# time step
t = np.arange(0, 10, 1e-3)
dt = t[1]

# Define variable
Q = np.array([[2, 0], [0, 0.01]])
R = 0.1
P1 = np.array([[1, 0], [0, 0.01]])
A = np.array([[0, 1], [-1.6, -0.4]])
B = np.array([[0], [1]])
P_10 = P1
x_10 = np.zeros((4, 1))
Arow1 = np.append(A, -B/R@B.T, axis=1)
Arow2 = np.append(-Q, -A.T, axis=1)
A_1 = np.append(Arow1, Arow2, axis=0)
y = np.zeros((2, len(t)))
# find P
P = rk4(P_10, t, dt, riccati)

# find TPBVP solution
sol = solve_bvp(TBVBP, bc, t, y, 0)

# simulation
X = np.zeros((2, len(t)))
u = np.zeros((1, len(t)))
X_initial = np.array([[10], [0]])
X[0:2, 0:1] = np.array([[10], [0]])


for i in range(0, len(t) - 1):
    u[:, i:i + 1] = -1 / R * B.T @ P[0:2, 2 * i:2 * i + 2] @ X[0:2, i:i + 1]
    X[:, i + 1:i + 2] = X[0:2, i:i + 1] + dt * (A @ X[0:2, i:i + 1] + B @ u[0:1, i:i + 1])


# two plot
fig, (ax1, ax2) = plt.subplots(1, 2)

# plot riccati
ax1.plot(X[0, :], X[1, :], label = 'riccati')
ax1.scatter(X[0, 0], X[1, 0], label = 'start point')
ax1.scatter(X[0, len(t) - 1], X[1, len(t) - 1], label = 'final point')
ax1.legend()
ax1.grid()

# plot TPBVP
ax2.plot(X2[0, :], X2[1, :], label='TPBVP')
ax2.scatter(X2[0, 0], X2[1, 0], label='start point')
ax2.scatter(X2[0, len(t) - 1], X2[1, len(t) - 1], label='final point')
ax2.legend()
ax2.grid()

# show fig
plt.show()
