import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, pow

# define system
def f(x, u):
    dx_dt = np.zeros((3, 1))
    dx_dt[0] = cos(x[2]) * u[0]
    dx_dt[1] = sin(x[2]) * u[0]
    dx_dt[2] = u[1]
    return dx_dt


# simulation
def sim(u, t, X_initial):
    X = np.zeros((3, len(t)))
    X[:, 0:1] = X_initial
    dt = t[1]
    for i in range(0, len(t)-1):
        X[:, i+1:i+2] = X[:, i:i+1] + dt * f(X[:, i:i+1], u[:, i])
    return X


# define Riccati equation
def ricatti(P, A, B, a, b, P_know, r_know):
    return -P @ A - A.T @ P + P @ B @ np.linalg.inv(R) @ B.T @ P - Q


# define r equation
def rdot(r, A, B, a, b, P_know, r_know):
    return -(A - B @ np.linalg.inv(R) @ B.T @ P_know).T @ r - a + P_know @ B @ np.linalg.inv(R) @ b


# define z equation
def zdot(z, A, B, a, b, P_know, r_know):
    return A @ z + B @ (-np.linalg.inv(R) @ B.T @ P_know @ z - np.linalg.inv(R) @ B.T @ r_know - np.linalg.inv(R) @ b)


# define v equation
def v1(z, A, B, a, b, P_know, r_know):
    return -np.linalg.inv(R) @ B.T @ P_know @ z - np.linalg.inv(R) @ B.T @ r_know - np.linalg.inv(R) @ b


# find cost function f
def J(x, desire, input):
    cost = 0
    for i in range(0, len(t)):
        cost = cost + 1/2*(x[:, i] - desire[:, i]).T @ Q @ (x[:, i] - desire[:, i]) + 1/2*input[:, i].T @ R @ input[:, i]
    cost = cost + 1/2*(x[:, len(t)-1] - desire[:, len(t)-1]).T @ P1 @ (x[:, len(t)-1] - desire[:, len(t)-1])
    return cost


# find a(t) and b(t)
def Dl(x, desire, input):
    a = np.zeros((3, len(t)))
    b = np.zeros((2, len(t)))
    for i in range(0, len(t)):
        a[:, i] = Q.T @ (x[:, i] - desire[:, i])
        b[:, i] = R.T @ input[:, i]
    return a, b


# define zeta norm
def zetanorm(zeta):
    norm = 0
    for i in range(0, len(t)):
        norm = norm + np.linalg.norm(zeta[:, i])
    return norm


# define A matrix
def Amatrix(zeta, u):
    A = np.zeros((3, 3 * len(t)))
    for i in range(0, len(t)):
        A[:, 3 * i:3 * i + 3] = np.array(
            [[0, 0, cos(zeta[2, i]) * u[0, i]], [0, 0, sin(zeta[2, i]) * u[0, i]], [0, 0, 0]])
    return A


# define B matrix
def Bmatrix(zeta, u):
    B = np.zeros((3, 2 * len(t)))
    for i in range(0, len(t)):
        B[:, 2 * i:2 * i + 2] = np.array([[cos(zeta[2, i]), 0], [sin(zeta[2, i]), 0], [0, 1]])
    return B


# Finds value of y for a given x using step size dt
# and initial value y0 at x0.
def rk4(sol_final, t, dt, f, A, B, a, b, P, r):
    # Iterate for number of iterations
    row = len(sol_final)
    columns = len(sol_final[0])
    sol = np.zeros((row, columns * len(t)))
    sol[:, (columns * len(t) - columns):(columns * len(t))] = sol_final
    for i in range(len(t) - 1, 0, -1):
        k1 = dt * f(sol[:, columns * i:(columns * i + columns)], A[:, 3 * i:3 * i + 3], B[:, 2 * i:2 * i + 2],
                    a[:, i:i + 1], b[:, i:i + 1], P[:, 3 * i:3 * i + 3], r[:, i:i + 1])
        k2 = dt * f(sol[:, columns * i:(columns * i + columns)] - k1 / 2., A[:, 3 * i:3 * i + 3], B[:, 2 * i:2 * i + 2],
                    a[:, i:i + 1], b[:, i:i + 1], P[:, 3 * i:3 * i + 3], r[:, i:i + 1])
        k3 = dt * f(sol[:, columns * i:(columns * i + columns)] - k2 / 2., A[:, 3 * i:3 * i + 3], B[:, 2 * i:2 * i + 2],
                    a[:, i:i + 1], b[:, i:i + 1], P[:, 3 * i:3 * i + 3], r[:, i:i + 1])
        k4 = dt * f(sol[:, columns * i:(columns * i + columns)] - k3, A[:, 3 * i:3 * i + 3], B[:, 2 * i:2 * i + 2],
                    a[:, i:i + 1], b[:, i:i + 1], P[:, 3 * i:3 * i + 3], r[:, i:i + 1])
        # Update next value of sol
        sol[:, (columns * i - columns):columns * i] = sol[:, columns * i:(columns * i + columns)] - (1 / 6.) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return sol


# time step
t = np.arange(0, 2 * pi, 1e-3)

# initial condition
X_initial = np.array([[0], [0], [pi / 2]])

# desire trajectory
x_d = 2 / pi * t
y_d = np.zeros(t.shape)
theta_d = pi / 2 * np.ones(t.shape)
desire = np.vstack((x_d, y_d, theta_d))

# initial input
u = np.vstack((1 * np.ones(t.shape), -1/2 * np.ones(t.shape)))
u_old = u

# initial simulation
x_0 = sim(u, t, X_initial)
x_old = x_0

# initial kxi
kxi = np.append(x_0, u, axis=0)

# Define LQR variable
Q = np.array([[2, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
R = np.array([[0.1, 0], [0, 0.1]])
P1 = np.array([[1, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

# Define cost function
cost = J(x_0, desire, u)


def iLQR(x_i, desire, u_i):

    # find a(t),b(t)
    a, b = Dl(x_i, desire, u_i)

    # Define system variable
    P_final = P1
    r_final = (x_i[:, len(t.shape) - 1:len(t.shape)] - desire[:, len(t.shape) - 1:len(t.shape)]).T @ P1
    dt = t[1]
    A = Amatrix(x_i, u_i)
    B = Bmatrix(x_i, u_i)
    P = np.zeros((3, 3 * len(t)))
    r = np.zeros((3, 1 * len(t)))
    z_final = np.zeros((3, 1))

    # find P
    P = rk4(P_final, t, dt, ricatti, A, B, a, b, P, r)

    # find r
    r = rk4(r_final.T, t, dt, rdot, A, B, a, b, P, r)

    # find z
    z = rk4(z_final, t, dt, zdot, A, B, a, b, P, r)

    # find v
    v = np.zeros((2, len(t)))
    for i in range(0, len(t)):
        v[:, i:i + 1] = v1(z[:, i: i + 1], A[:, 3 * i: 3 * i + 3], B[:, 2 * i:2 * i + 2], a[:, i:i + 1], b[:, i:i + 1],
                           P[:, 3 * i:3 * i + 3], r[:, i: i + 1])

    # find gradient descent
    zeta = np.append(z, v, axis=0)

    # find DJ(kxi) dot zeta
    DJdotzeta = 0
    for i in range(0, len(t)):
        DJdotzeta = DJdotzeta + np.dot(a[:, i], z[:, i]) + np.dot(b[:, i], v[:, i])

    # initial  Armijo line search coefficient
    alpha = 0.3
    beta = 0.5
    N = 1

    # Armijo line search
    while True:
        gamma = pow(beta, N)
        u_new = u_i + gamma * v
        x_new = sim(u_new, t, X_initial)
        # compare current cost function with linear
        if J(x_new, desire, u_new) <= J(x_0, desire, u)+alpha*gamma*DJdotzeta:
            break
        N = N + 1

    print(zetanorm(zeta))
    return x_new, u_new, z, v


# iterative
for i in range(0, 5):
    x_new, u_new, z, v = iLQR(x_old, desire, u_old)
    x_old = x_new
    u_old = u_new

# plot initial guess and reference
plt.plot(x_0[0, :], x_0[1, :], label = 'initial')
plt.plot(x_new[0, :], x_new[1, :], label = 'first iteration')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



