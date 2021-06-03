import numpy as np
from math import pi, cos, sin, pow
import matplotlib.pyplot as plt


def integrate(f, xt, dt, tt):
    """
    This function takes in an initial condition x(t) and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x(t). It outputs a vector x(t+dt) at the future
    time step.

    Parameters
    ============
    dyn: Python function
        derivate of the system at a given step x(t),
        it can considered as \dot{x}(t) = func(x(t))
    xt: NumPy array
        current step x(t)
    dt:
        step size for integration
    tt:
        current time

    Return
    ============
    new_xt:
        value of x(t+dt) integrated from x(t)
    """
    k1 = dt * f(xt, tt)
    k2 = dt * f(xt + k1 / 2., tt + dt / 2.)
    k3 = dt * f(xt + k2 / 2., tt + dt / 2.)
    k4 = dt * f(xt + k3, tt + dt)
    new_xt = xt + (1 / 6.) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new_xt


def simulate(f, x0, tspan, dt, integrate):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).

    Parameters
    ============
    f: Python function
        derivate of the system at a given step x(t),
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        initial conditions
    tspan: Python list
        tspan = [min_time, max_time], it defines the start and end
        time of simulation
    dt:
        time step for numerical integration
    integrate: Python function
        numerical integration method used in this simulation

    Return
    ============
    x_traj:
        simulated trajectory of x(t) from t=0 to tf
    """
    N = int((max(tspan) - min(tspan)) / dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan), max(tspan), N)
    xtraj = np.zeros((len(x0), N))
    for i in range(N):
        xtraj[:, i] = integrate(f, [*x], dt, tvec[i])
        x = np.copy(xtraj[:, i])
    return xtraj


# define system
def f(xu, tt):
    try:
        dx_dt = np.array([cos(xu[2]) * u[0, int(tt/dt)], sin(xu[2]) * u[0, int(tt/dt)], u[1, int(tt/dt)]])
    except:
        dx_dt = np.array([cos(xu[2]) * u[0, len(t)-1], sin(xu[2]) * u[0, len(t)-1], u[1, len(t)-1]])
    return dx_dt


# define cost function l
def l(J, tt):
    try:
        cost = (kxi[:, int(tt/dt)]-desire[:, int(tt/dt)]).T @ Q @ (kxi[:, int(tt/dt)] - desire[:, int(tt/dt)]) + u[:, int(tt/dt)].T @ R @ u[:, int(tt/dt)]
    except:
        cost = 0
    return cost


# define A matrix
def A(tt):
    try:
        return np.array([[0, 0, -sin(kxi[2, int(tt/dt)]) * u[0, int(tt/dt)]], [0, 0, cos(kxi[2, int(tt/dt)]) * u[0, int(tt/dt)]], [0, 0, 0]])
    except:
        return np.array([[0, 0, -sin(kxi[2, len(kxi[0])-1]) * u[0, len(kxi[0])-1]], [0, 0, cos(kxi[2, len(kxi[0])-1]) * u[0, len(kxi[0])-1]], [0, 0, 0]])


# define B matrix
def B(tt):
    try:
        return np.array([[cos(kxi[2, int(tt/dt)]), 0], [sin(kxi[2, int(tt/dt)]), 0], [0, 1]])
    except:
        return np.array([[cos(kxi[2, len(kxi[0])-1]), 0], [sin(kxi[2, len(kxi[0])-1]), 0], [0, 1]])


def a(tt):
    try:
        return (kxi[:, int(tt/dt)]-desire[:, int(tt/dt)]).T @ Q
    except:
        return (kxi[:, len(kxi[0])-1]-desire[:, len(kxi[0])-1]).T @ Q


def b(tt):
    try:
        return u[:, int(tt/dt)].T @ R
    except:
        return u[:, len(kxi[0])-1].T @ R


# define Riccati equation
def riccati(P, tt):
    return -P @ A(tt) - A(tt).T @ P + P @ B(tt) @ np.linalg.inv(R) @ B(tt).T @ P - Q


# define r equation
def rdot(r, tt):
    return -(A(tt) - B(tt) @ np.linalg.inv(R) @ B(tt).T @ P[:, 3*int(tt/dt):3*int(tt/dt)+3]).T @ r - a(tt).T + P[:, 3*int(tt/dt):3*int(tt/dt)+3] @ B(tt) @ np.linalg.inv(R) @ b(tt).T


# define z equation
def zdot(z, tt):
    return A(tt) @ z + B(tt) @ v1(z, tt)


# define v equation
def v1(z, tt):
    try:
        return -np.linalg.inv(R) @ B(tt).T @ P[:, 3*int(tt/dt):3*int(tt/dt)+3] @ z - np.linalg.inv(R) @ B(tt).T @ r[:, int(tt/dt)] - np.linalg.inv(R) @ b(tt).T
    except:
        return -np.linalg.inv(R) @ B(tt).T @ P[:, 3 * (len(kxi[0])-1):3 * (len(kxi[0])-1) + 3] @ z - np.linalg.inv(R) @ B(tt).T @ r[:, len(kxi[0])-1] - np.linalg.inv(R) @ b(tt).T

# define directional diff.
def DiffJdotzeta(Dz, tt):
    try:
        return a(tt) @ z[:, int(tt/dt)] + b(tt) @ v[:, int(tt/dt)]
    except:
        return a(tt) @ z[:, len(kxi[0])-1] + b(tt) @ v[:, len(kxi[0])-1]


# define zeta norm
def zetanorm(zn, tt):
    try:
        return 0.5*z[:, int(tt/dt)].T @ Q @ z[:, int(tt/dt)] + 0.5*v[:, int(tt/dt)].T @ R @ v[:, int(tt/dt)]
    except:
        return 0


# time step
t = np.arange(0, 2 * pi, 1e-3)
dt = t[1]
tspan = np.array([t[0], t[len(t)-1]])

# initial value
kxi_0 = np.array([0, 0, pi / 2])
Q = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0.01]])
R = np.array([[0.01, 0], [0, 0.01]])
P1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

# input
u = np.vstack((1 * np.ones(len(t)), -1/2 * np.ones(len(t))))
u_initial = u


# initial simulation
kxi = simulate(f, kxi_0, tspan, dt, integrate)
kxi = np.append(kxi_0.reshape((3, 1)), kxi, axis=1)

# store initial simulation
kxi_initial = kxi

# desire trajectory
x_d = 2 / pi * t
y_d = np.zeros(t.shape)
theta_d = pi / 2 * np.ones(t.shape)
desire = np.vstack((x_d, y_d, theta_d))

for j in range(0, 5):

    # calculate cost function
    J = 0
    for i in range(0, len(kxi[0])):
        J = integrate(l, J, dt, t[i])
    print('J')
    print(J)

    # solve P
    P = np.zeros((3, 3 * len(kxi[0])))
    P[:, 3*len(kxi[0])-3:3*len(kxi[0])] = P1
    for i in range(len(kxi[0])-1, 0, -1):
        P[:, 3*i-3: 3*i] = integrate(riccati, P[:, 3*i: 3*i+3], -dt, t[i])

    # solve r
    r = np.zeros((3, len(kxi[0])))
    r[:, len(kxi[0])-1] = P1.T@(kxi[:, len(kxi[0])-1]-desire[:, len(kxi[0])-1])
    for i in range(len(kxi[0])-1, 0, -1):
        r[:, i-1] = integrate(rdot, r[:, i], -dt, t[i])

    # simulation z
    z = np.zeros((3, len(kxi[0])))
    z_0 = np.array([0, 0, 0])
    z = simulate(zdot, z_0, tspan, dt, integrate)
    z = np.append(z_0.reshape((3, 1)), z, axis=1)

    # solve v
    v = np.zeros((2, len(kxi[0])))
    for i in range(0, len(kxi[0])):
        v[:, i] = v1(z[:, i], t[i])

    # DJ(kxi) dot zeta
    DJdotz = 0
    for i in range(0, len(kxi[0])):
        DJdotz = integrate(DiffJdotzeta, DJdotz, dt, t[i])
    '''print('DJdotz')
    print(DJdotz)'''
    # initial Armijo line search coefficient
    alpha = 0.45
    beta = 0.2
    N = 0

    # store current input
    u_old = u

    # Armijo line search
    while True:
        gamma = pow(beta, N)
        u = u + gamma * v
        kxi = simulate(f, kxi_0, tspan, dt, integrate)
        kxi = np.append(kxi_0.reshape((3, 1)), kxi, axis=1)
        J_new = 0
        for i in range(0, len(kxi[0])):
            J_new = integrate(l, J_new, dt, t[i])
        '''print('J_new')
        print(J_new)'''

        # compare current cost function with linear
        if J_new <= J + alpha*gamma*DJdotz:
            break

        N = N + 1

        # reset input
        u = u_old

    # calculate zeta norm
    zetaN = 0
    for i in range(0, len(kxi[0])):
        zetaN = integrate(zetanorm, zetaN, dt, t[i])
    print('zetaN')
    print(zetaN)

    print(str(j+1) + 'loop complete')


# plot initial guess and reference
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(kxi_initial[0, :], kxi_initial[1, :], label='initial')
ax1.plot(kxi[0, :], kxi[1, :], label='iterative')
ax1.plot(desire[0, :], desire[1, :], label='desire')
ax1.legend()

ax2.plot(t, u_initial[0, :], label='initial input 1')
ax2.plot(t, u[0, :], label='new input 1')
ax2.plot(t, u_initial[1, :], label='initial input 2')
ax2.plot(t, u[1, :], label='new input 2')
ax2.legend()

plt.grid
plt.show()

