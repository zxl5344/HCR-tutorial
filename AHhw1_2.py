import numpy as np
import matplotlib.pyplot as plt


# define function
def f(P, A, B, Q, R):
    return -P * A - A.T * P + P * B / R * B.T * P + Q


# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def recatti(P_final, A, B, Q, R, t_final, dt):
    # Iterate for number of iterations
    P = np.zeros((2, 2 * int(t_final / dt)))
    P[0:2, (2 * int(t_final / dt) - 2):(2 * int(t_final / dt))] = P_final
    for i in range(int(t_final / dt) - 1, 0, -1):
        k1 = dt * f(P[0:2, 2 * i:2 * i + 2], A, B, Q, R)
        k2 = dt * f(P[0:2, 2 * i:2 * i + 2] + k1 / 2., A, B, Q, R)
        k3 = dt * f(P[0:2, 2 * i:2 * i + 2] + k2 / 2., A, B, Q, R)
        k4 = dt * f(P[0:2, 2 * i:2 * i + 2] + k3, A, B, Q, R)
        # Update next value of p
        P[0:2, 2 * i - 2:2 * i] = P[0:2, 2 * i:2 * i + 2] - (1 / 6.) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        '''
        print("old one")
        print(P[0:2, 2 * i:2 * i + 2])
        print('new one')
        print(P[0:2, 2*i-2:2*i])
        '''
    return P


# Define variable
Q = np.array([[2, 0], [0, 0.01]])
R = 0.1
P1 = np.array([[1, 0], [0, 0.01]])
A = np.array([[0, 1], [-1.6, -0.4]])
B = np.array([[0], [1]])
P_10 = P1
t_final = 10
dt = 0.001

# find P
P = recatti(P_10, A, B, Q, R, t_final, dt)

# simulation
X = np.zeros((2, int(t_final / dt)))
u = np.zeros((1, int(t_final / dt)))
X[0:2, 0:1] = np.array([[10], [0]])

for i in range(0, int(t_final / dt) - 1):
    u[0:1, i:i + 1] = -1 / R * B.T @ P[0:2, 2 * i:2 * i + 2] @ X[0:2, i:i + 1]
    X[0:2, i + 1:i + 2] = X[0:2, i:i + 1] + dt * (A @ X[0:2, i:i + 1] + B @ u[0:1, i:i + 1])

# plot
plt.plot(X[0, :], X[1, :])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
