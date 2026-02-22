import numpy as np
import scipy.sparse as sp
from euler_methods import approx_wave
from utils import plot, animate_wave, largerPlotFont

largerPlotFont()

def main():
    c = 1
    L = 1
    N = 400

    dx = L / N
    dt = .001

    x = np.arange(N + 1) * dx
    initConditions = [
        np.sin(2 * np.pi * x),
        np.sin(5 * np.pi * x),
        np.sin(5 * np.pi * x)
    ]
    initConditions[2] *= (x > .2) & (x < .4)

    D_2 = sp.diags((np.ones(N - 2), -2 * np.ones(N - 1), np.ones(N - 2)), (-1, 0, 1))

    Psi = approx_wave(initConditions[2], time_steps=150,
                      equations_matrix=D_2.toarray(),
                      dt=dt, c=c, dx=dx, N=N, method='SV')

    plot(Psi, x=x, N_steps=5, dt=dt)

    # animate_wave(Psi,x=x, dt=dt)

if __name__ == '__main__':
    main()

# def animate(Psi_0):
#     fig, ax = plt.subplots()
#
#     frame_step = 5
#
#     line = ax.plot(x, Psi_0)[0]
#     ax.set_xlabel("$x$")
#     ax.set_ylabel("$\\Psi(x, t)$")
#     ax.set_ylim(-np.max(np.abs(Psi_0)), np.max(np.abs(Psi_0)))
#
#     Psi_now = Psi_0[1:N]
#     Psi_prev = Psi_0[1:N]
#
#     # Update function for animation
#     def update(frame):
#         nonlocal Psi_now, Psi_prev
#
#         for _ in range(frame_step):
#             Psi_next = 2 * Psi_now - Psi_prev + c * (dt / dx) ** 2 * D_2.dot(Psi_now)
#
#             Psi_prev = Psi_now
#             Psi_now = Psi_next
#
#         Psi_plot = np.insert(Psi_now, (0, N - 1), 0)
#         line.set_ydata(Psi_plot)
#         ax.set_title("t = " + str(frame * 5 * dt)[:4])
#
#         return
#
#     ani = animation.FuncAnimation(
#         fig, update, interval=10, cache_frame_data=False
#     )
#
#     plt.show()
#
#     return
