import numpy as np
from typing import Any, Union
import numpy.typing as npt


def approx_wave(psi_0:npt.NDArray, time_steps: Union[int, np.integer[Any]],
                mesh_points: npt.NDArray,
                equations_matrix: npt.NDArray,
                dt: Union[int, float],
                c: Union[int,float],
                dx: Union[int,float],
                N:int,
                method:str ="EF") -> npt.NDArray:

    psi_0 = psi_0[1:N]
    psi = np.empty((N-1,time_steps+1), dtype=np.float64)
    y = np.empty((N-1,time_steps+1), dtype=np.float64)

    psi[:,0] = psi_0
    y[:,0] = np.zeros(N-1,dtype=np.float64)

    if method.upper() == "EF":
        # Euler forward method
        for niter in range(time_steps):
            d2psi_dx2 = equations_matrix.dot(psi[:,niter]) / dx**2

            psi[:,niter+1] = psi[:,niter] +  dt * y[:,niter]
            y[:,niter+1] = y[:,niter] + dt * c**2 * d2psi_dx2

    elif method.upper() == "RK":
        # Runge-Kutta 4
        for niter in range(time_steps):
            k1_psi = y[:,niter]
            k1_y = c ** 2 * equations_matrix.dot(psi[:,niter]) / dx ** 2

            k2_psi = y[:,niter] + 0.5 * dt * k1_y
            k2_y = c ** 2 * equations_matrix.dot(psi[:,niter] + 0.5 * dt * k1_psi) / dx ** 2

            k3_psi = y[:,niter] + 0.5 * dt * k2_y
            k3_y = c ** 2 * equations_matrix.dot(psi[:,niter] + 0.5 * dt * k2_psi) / dx ** 2

            k4_psi = y[:,niter] + dt * k3_y
            k4_y = c ** 2 * equations_matrix.dot(psi[:,niter] + dt * k3_psi) / dx ** 2

            psi[:, niter + 1] = psi[:,niter] + dt / 6 * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
            y[:, niter + 1] = y[:,niter] + dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

    else:
        raise ValueError("Invalid method, must be 'EF' or 'RK'")

    psi = psi[:,1:]
    psi = np.insert(psi, (0,N - 1), np.zeros(time_steps),axis=0)

    return psi