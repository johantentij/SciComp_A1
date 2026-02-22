import numpy as np
from typing import Any, Union
import numpy.typing as npt
from numba import njit
import numexpr as ne


@njit
def __euler_forward_1D(psi, y, equations_matrix, dx, dt, c, time_steps):
    for niter in range(time_steps):
        d2psi_dx2 = equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2

        psi[:, niter + 1] = psi[:, niter] + dt * y[:, niter]
        y[:, niter + 1] = y[:, niter] + dt * c ** 2 * d2psi_dx2

@njit
def __euler_forward_2D(c_0, time_steps, dt, D, dx, N):
    constant = D * dt / dx ** 2

    for niter in range(time_steps):
        # Current state (2D slice)
        ck = c_0[:, :, niter]

        # We iterate through the rows to handle the periodic wrap manually
        for i in range(N ):
            # Periodic indices for the first dimension
            right_roll = (i + 1) % (N)
            right_roll = (i - 1) % (N)

            # Update the inner region (1 to N-1) of the second dimension
            c_0[i, 1:N, niter + 1] = (
                    ck[i, 1:N] + constant * (
                    ck[right_roll, 1:N] +  # c_right equivalent
                    ck[right_roll, 1:N] +  # c_left equivalent
                    ck[i, 2:N + 1] +  # c_up (j+1)
                    ck[i, 0:N - 1] -  # c_down (j-1)
                    4 * ck[i, 1:N]  # central point
            ))
    return c_0

@njit
def __runge_kutta4(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Runge-Kutta 4
    for niter in range(time_steps):
        k1_psi = y[:, niter]
        k1_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2

        k2_psi = y[:, niter] + 0.5 * dt * k1_y
        k2_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + 0.5 * dt * k1_psi) / dx ** 2

        k3_psi = y[:, niter] + 0.5 * dt * k2_y
        k3_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + 0.5 * dt * k2_psi) / dx ** 2

        k4_psi = y[:, niter] + dt * k3_y
        k4_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + dt * k3_psi) / dx ** 2

        psi[:, niter + 1] = psi[:, niter] + dt / 6 * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
        y[:, niter + 1] = y[:, niter] + dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

@njit
def __stormer_verlet(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Størmer–Verlet method
    psi[:, 1] = psi[:,0]
    for niter in range(2, time_steps + 1):
        psi[:, niter] = (2 * psi[:, niter - 1] - psi[:, niter - 2] +
                         c * (dt / dx) ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter - 1])))

@njit
def __leapfrog(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Leapfrog method
    for niter in range(time_steps):
        d2psi_dx2 = equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2
        psi[:, niter + 1] = psi[:, niter] + dt * y[:, niter] + 0.5 * c ** 2 * d2psi_dx2 * dt ** 2

        d2psi_dx2_next = equations_matrix.dot(np.ascontiguousarray(psi[:, niter + 1])) / dx ** 2
        y[:, niter + 1] = y[:, niter] + 0.5 * (d2psi_dx2 + d2psi_dx2_next) * dt

##
## Parallel version of the above functions
##

def __euler_forward_1D_paral(psi, y, equations_matrix, dx, dt, c, time_steps):
    for niter in range(time_steps):
        d2psi_dx2 = equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2

        ## Needed for numexpr package as it does not understand pythonic array splicing
        psi_n = psi[:,niter]
        y_n = y[:,niter]

        psi[:, niter + 1] = ne.evaluate('psi_n + dt * y_n')
        y[:, niter + 1] = ne.evaluate('y_n + dt * c ** 2 * d2psi_dx2')

def __euler_forward_2D_paral(c_0, time_steps, dt, D, dx, N):
    constant = D * dt / dx**2

    for niter in range(time_steps):
        ck = c_0[:, :, niter]

        c_right = np.roll(ck, -1, axis=0)  # c[i+1, j, k]  (periodic wrap)
        c_left = np.roll(ck, 1, axis=0)  # c[i-1, j, k]  (periodic wrap)

        c_current = ck[:, 1:N]
        c_right_aux = c_right[:, 1:N]
        c_left_aux = c_left[:, 1:N]
        c_up = ck[:, 2:N + 1]
        c_down = ck[:, 0:N - 1]

        c_0[:, 1:N, niter + 1] = ne.evaluate('c_current + constant * (c_right_aux \
        + c_left_aux + c_up  + c_down - 4 * c_current)')

def __runge_kutta4_paral(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Runge-Kutta 4
    for niter in range(time_steps):

        ## Needed for numexpr package as it does not understand pythonic array splicing
        psi_n = psi[:, niter]
        y_n = y[:, niter]

        k1_psi = y[:, niter]
        k1_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2

        k2_psi = y[:, niter] + 0.5 * dt * k1_y
        k2_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + 0.5 * dt * k1_psi) / dx ** 2

        k3_psi = y[:, niter] + 0.5 * dt * k2_y
        k3_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + 0.5 * dt * k2_psi) / dx ** 2

        k4_psi = y[:, niter] + dt * k3_y
        k4_y = c ** 2 * equations_matrix.dot(np.ascontiguousarray(psi[:, niter]) + dt * k3_psi) / dx ** 2

        psi[:, niter + 1] = ne.evaluate('psi_n + dt / 6 * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)')
        y[:, niter + 1] = ne.evaluate('y_n + dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)')

def __stormer_verlet_paral(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Størmer–Verlet method
    psi[:, 1] = psi[:,0]
    for niter in range(2, time_steps + 1):
        d2psi_dx2 = equations_matrix.dot(np.ascontiguousarray(psi[:, niter - 1]))

        ## Needed for numexpr package as it does not understand pythonic array splicing
        psi_n = psi[:, niter - 1]
        psi_n_prev = psi[:, niter - 2]

        psi[:, niter] = ne.evaluate('2 * psi_n - psi_n_prev + \
                         c * (dt / dx) ** 2 * d2psi_dx2')

def __leapfrog_paral(psi, y, equations_matrix, dx, dt, c, time_steps):
    # Leapfrog method
    for niter in range(time_steps):
        d2psi_dx2 = equations_matrix.dot(np.ascontiguousarray(psi[:, niter])) / dx ** 2

        ## Needed for numexpr package as it does not understand pythonic array splicing
        y_n = y[:,niter]
        psi_n = psi[:,niter]

        psi[:, niter + 1] = ne.evaluate('psi_n + dt * y_n + 0.5 * c ** 2 * d2psi_dx2 * dt ** 2')

        d2psi_dx2_next = equations_matrix.dot(np.ascontiguousarray(psi[:, niter + 1])) / dx ** 2
        y[:, niter + 1] = ne.evaluate('y_n + 0.5 * (d2psi_dx2 + d2psi_dx2_next) * dt')

APPROX_METHOD = {
    "EF": __euler_forward_1D,
    "RK4": __runge_kutta4,
    "LP": __leapfrog,
    "SV": __stormer_verlet,
}

APPROX_METHOD_PARALLEL = {
    "EF": __euler_forward_1D_paral,
    "RK4": __runge_kutta4_paral,
    "LP": __leapfrog_paral,
    "SV": __stormer_verlet_paral,
}

APPROX_METHOD_HEAT ={
    "EF": __euler_forward_2D
}

APPROX_METHOD_HEAT_PARALLEL = {
    "EF": __euler_forward_2D_paral,
}

def approx_wave(psi_0:npt.NDArray, time_steps: Union[int, np.integer[Any]],
                equations_matrix: npt.NDArray,
                dt: Union[int, float],
                c: Union[int,float],
                dx: Union[int,float],
                N: int,
                method: str="EF",
                threads: int=None) -> npt.NDArray:

    if threads:
        operation = APPROX_METHOD_PARALLEL.get(method.upper())
        ne.set_num_threads(threads)
    else:
        operation = APPROX_METHOD.get(method.upper())

    if operation:
        psi_0 = psi_0[1:N]
        psi = np.empty((N-1,time_steps+1), dtype=np.float64)
        psi[:, 0] = psi_0
        y = None

        if method.upper() != "SV":
            y = np.empty((N-1,time_steps+1), dtype=np.float64)
            y[:,0] = np.zeros(N-1,dtype=np.float64)

        # Function to generate the approximation
        operation(psi, y, equations_matrix, dx, dt, c, time_steps)

        psi = psi[:, 1:]
        psi = np.insert(psi, (0, N - 1), np.zeros(time_steps), axis=0)

        return psi

    else:
        raise ValueError(f"Invalid method, must be {list(APPROX_METHOD.keys())}")

def approx_heat(c_0:npt.NDArray, time_steps: Union[int, np.integer[Any]],
                dt: Union[int, float],
                D: Union[int,float],
                dx: Union[int,float],
                N: int,
                method: str="EF",
                threads: int=None):

    if threads:
        operation = APPROX_METHOD_HEAT_PARALLEL.get(method.upper())
        ne.set_num_threads(threads)
    else:
        operation = APPROX_METHOD_HEAT.get(method.upper())

    if operation:
        operation(c_0, time_steps, dt, D, dx, N)
        return c_0

    else:
        raise ValueError(f"Invalid method, must be {list(APPROX_METHOD.keys())}")
