import numpy as np
import matplotlib.pyplot as plt
from numba import njit

D = 1
N = 100

dx = 1 / N

c = np.zeros((N, N + 1), dtype=np.float64)
c[:, N] = 1

def Jacobi(c, N_iter=10000):
    for _ in range(N_iter):
        c_old = np.copy(c)

        c_left = np.roll(c_old, 1, axis=0)
        c_right = np.roll(c_old, -1, axis=0)

        c[:, 1:N] = .25 * (
            c_left[:, 1:N] + 
            c_right[:, 1:N] +
            c_old[:, 0:N-1] + 
            c_old[:, 2:N+1]
        )

    return c

@njit
def Gauss_Seidel(c, N_iter=5000):
    for _ in range(N_iter):
        for y in range(1, N):
            y_down = y - 1
            y_up = y + 1

            for x in range(N):
                x_left = (x - 1) % N
                x_right = (x + 1) % N

                c[x, y] = .25 * (
                    c[x_left, y] + 
                    c[x_right, y] + 
                    c[x, y_up] + 
                    c[x, y_down]
                )

    return c

@njit
def SOR(c, omega=1.8, N_iter=500):
    for _ in range(N_iter):
        for y in range(1, N):
            y_down = y - 1
            y_up = y + 1

            for x in range(N):
                x_left = (x - 1) % N
                x_right = (x + 1) % N

                c[x, y] = .25 * omega * (
                    c[x_left, y] + 
                    c[x_right, y] + 
                    c[x, y_up] + 
                    c[x, y_down]
                ) + (1 - omega) * c[x, y]

    return c

x = np.arange(N) * dx
y = np.arange(N + 1) * dx

c_solved = SOR(c)

# plt.pcolor(x, y, c_solved.T)
plt.plot(y, c_solved[0, :])
plt.show()





