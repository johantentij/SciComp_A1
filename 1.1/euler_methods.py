import numpy as np
from typing import Any, Union
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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


def animate_wave(psi, x, dt, frame_skip=5, save_path=None):
    """
    Animate the wave solution.

    Parameters:
    -----------
    psi : ndarray, shape (N+1, time_steps)
        Solution array from approx_wave
    x : ndarray, shape (N+1,)
        Spatial mesh points
    dt : float
        Time step
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'wave.mp4' or 'wave.gif')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initial plot
    line, = ax.plot(x, psi[:, 0], 'b-', linewidth=2)

    # Set fixed axis limits
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(psi) * 1.1, np.max(psi) * 1.1)

    ax.set_xlabel('Position (x)', fontsize=12)
    ax.set_ylabel('Wave amplitude (Ψ)', fontsize=12)
    ax.set_title('Wave Equation Solution', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')

    def init():
        """Initialize animation"""
        line.set_data(x, psi[:, 0])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        """Update function for each frame"""
        line.set_data(x, psi[:, frame])
        time_text.set_text(f'Time = {frame * dt:.3f}')
        return line, time_text

    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=range(0, psi.shape[1], frame_skip),
        init_func=init, interval=10
    )

    # Save if path provided
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=30)
        print(f"Animation saved to {save_path}")

    plt.show()
    return anim


