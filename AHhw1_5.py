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
    # print(tt+dt)
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
def f1(x, tt):
    dx_dt = np.array([cos(x[2]) * u1[0, int(tt / dt)], sin(x[2]) * u1[0, int(tt / dt)], u1[1, int(tt / dt)]])
    return dx_dt


# define Dl1(tt)
def Dl1(tt):
    return (x[:, int(tt / dt)] - desire[:, int(tt / dt)]).T @ Q


# define A matrix
def A(tt):
    return np.array([[0, 0, -sin(x[2, int(tt / dt)]) * u1[0, int(tt / dt)]],
                     [0, 0, cos(x[2, int(tt / dt)]) * u1[0, int(tt / dt)]], [0, 0, 0]])


# define h matrix
def h(tt):
    return np.array([[cos(x[2, int(tt/dt)]), 0], [sin(x[2, int(tt/dt)]), 0], [0, 1]])


# define rhodot
def rhodot(rho, tt):
    return -A(tt).T @ rho - Dl1(tt)


# define cost function
def l(J1, tt):
    cost = (x[:, int(tt / dt)] - desire[:, int(tt / dt)]).T @ Q @ \
           (x[:, int(tt / dt)] - desire[:, int(tt / dt)]) + u1[:, int(tt / dt)].T @ R @ u1[:, int(tt / dt)]
    return cost


# define lambda
def Lambda(tt):
    return h(tt).T @ rho[:, int(tt / dt):(int(tt / dt)+1)] @ rho[:, int(tt / dt):(int(tt / dt)+1)].T @ h(tt)


# define u2*
def u2star(tt):
    return np.linalg.inv(Lambda(tt) + R.T) @ (Lambda(tt) @ u1[:, int(tt / dt)] + h(tt).T @ rho[:, int(tt / dt)] * alpha_d)


# calculate dJ1_dlambda
def dJ1_dlambda(tt):
    return rho[:, int(tt / dt)].T @ (h(tt) @ u2star(tt) - h(tt) @ u1[:, int(tt / dt)])


# second cost function J_tao
def J_tao(tt):
    return np.linalg.norm(u2star(tt)) + dJ1_dlambda(tt) + pow(tt-t_0, beta)


# define all constant

J_min = 0.0001  # minimum change in cost
t_curr = 0  # current time
t_init = 0.1  # default control duration
omega = 0.5  # scale factor
T = 2.  # predictive horizon
ts = 0.01  # sampling time
t_calc = 0.  # the max time for iterative control calculation
k_max = 10  # the max backtracking iterations
gamma = -9.  # the coefficient for first order sensitivity alpha
beta = 0.7
Q = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0.01]])  # Q in cost function
R = np.array([[0.01, 0], [0, 0.01]])  # R in cost function

# the end time t_end
t_end = 2*pi

# define x_init
x_init = np.array([0, 0, pi / 2])

while t_curr < t_end:

    # define t_0 and t_final
    t_0 = 0.
    t_final = T
    t = np.arange(t_0, t_final + ts, ts)
    tspan = np.array([t_0, t_final])
    dt = ts

    # define td for desire trajectory
    td = np.array([i+t_curr for i in t])

    # desire trajectory
    x_d = 2 / pi * td
    y_d = np.zeros(td.shape)
    theta_d = pi / 2 * np.ones(td.shape)
    desire = np.vstack((x_d, y_d, theta_d))
    #print('desire.shape')
    #print(desire.shape)

    # define nominal input
    if t_curr == 0:
        u1 = np.vstack((0.1 * np.ones(len(t)), -0.05 * np.ones(len(t))))
    else:
        u1_new = np.array([[1], [-0.5]])
        u1 = np.append(u1[:, 1:len(t)], u1_new, axis=1)
    #print('u1.shape')
    #print(u1.shape)

    # solve x
    x = simulate(f1, x_init, tspan, ts, integrate)
    x = np.append(x_init.reshape((3, 1)), x, axis=1)
    #print('x.shape')
    #print(x.shape)

    # solve rho
    rho = np.zeros((3, len(t)))
    rho[:, len(t) - 1] = np.zeros(3)
    for i in range(len(t) - 1, 0, -1):
        rho[:, i - 1] = integrate(rhodot, rho[:, i], -dt, t[i])
    #print('rho.shape')
    #print(rho.shape)

    # calculate cost function
    J_init = 0
    for i in range(0, len(t)):
        J_init = integrate(l, J_init, dt, t[i])
    print('J')
    print(J_init)

    # calculate sensitivity alpha
    alpha_d = gamma * J_init

    # find tao when u2star(tao) is min
    tao = t_0+t_calc
    minJtao = J_tao(t_0+t_calc)
    for i in range(0, len(x[0])):
        if t[i] > t_0+t_calc:
            if J_tao(t[i]) < minJtao:
                minJtao = J_tao(t[i])
                tao = t[i]
    print('tao')
    print(tao)

    # initial k and J1_new
    k = 0
    J_new = 10e8

    # store old u1
    u1_old = u1

    # search for lambda
    while J_new - J_init > J_min and k <= k_max:

        # restore the data
        u1 = u1_old

        # calculate lambda, tao_0 & tao_f
        lambda1 = (omega**k)*t_init
        tao_0 = tao - lambda1/2
        if tao_0 < 0:
            tao_0 = 0
        tao_f = tao + lambda1/2
        if tao_f > t_final:
            tao_f = t_final
        #print('lambda1, tao_0, tao_f')
        #print(lambda1, tao_0, tao_f)

        # add u2star to u1 and saturate
        for i in range(int(tao_0/dt), int(tao_f/dt)):

            # saturate u2star
            u2_new = u2star(i*dt)
            if u2_new[0] > 5:
                u1[0, i] = 5
            elif u2_new[0] < -5:
                u1[0, i] = -5
            else:
                u1[0, i] = u2_new[0]

            if u2_new[1] > 5:
                u1[1, i] = 5
            elif u2_new[1] < -5:
                u1[1, i] = -5
            else:
                u1[1, i] = u2_new[1]

        # re_simulation
        x = simulate(f1, x_init, tspan, dt, integrate)
        x = np.append(x_init.reshape((3, 1)), x, axis=1)

        # calculate new cost
        J_new = 0
        for i in range(0, len(x[0])):
            J_new = integrate(l, J_new, dt, t[i])
        k = k+1
        #print('J_new')
        #print(J_new)

    # store desire trajectory and real trajectory
    if t_curr == 0:
        X_d = desire[:, 0:1]
        X = x[:, 0:1]
        U = u1[:, 0:1]
    else:
        X_d = np.append(X_d, desire[:, 0:1], axis=1)
        X = np.append(X, x[:, 0:1], axis=1)
        U = np.append(U, u1[:, 0:1], axis=1)

    # update time and x_init
    t_curr = t_curr + ts
    x_init = x[:, int(ts/ts)]

    print('current time ' + str(t_curr))

# simulation with update input
x_init1 = np.array([0, 0, pi / 2])
u1 = U
tspan1 = np.array([0, int(2*pi)])
X1 = simulate(f1, x_init1, tspan1, dt, integrate)
X1 = np.append(x_init1.reshape((3, 1)), X1, axis=1)

# plot initial
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(X1[0, :], X1[1, :], label='update')
ax1.plot(X_d[0, :], X_d[1, :], label='desire')
ax1.legend()

ax2.plot(X[0, :], X[1, :])

ax3.plot(U[0, :], label='u1')
ax3.plot(U[1, :], label='u2')
ax3.legend()

plt.show()
