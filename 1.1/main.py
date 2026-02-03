import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from euler_methods import approx_wave

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

def integrate(Psi_0, T_steps=100):
    Psi_0 = Psi_0[1:N]

    Psi = np.empty((N - 1, T_steps + 1))
    Psi[:, 0] = Psi_0
    Psi[:, 1] = Psi_0

    for i in range(2, T_steps + 1):
        Psi[:, i] = 2 * Psi[:, i - 1] - Psi[:, i - 2] + c * (dt / dx) ** 2 * D_2.dot(Psi[:, i - 1])

    Psi = Psi[:, 1:]

    # add zeroes for completeness
    Psi = np.insert(Psi, (0, N - 1), np.zeros(T_steps), axis=0)

    return Psi

def plot(Psi, N_steps=10):
    T_steps = np.shape(Psi)[1]

    t_vals = np.linspace(0, T_steps * dt, N_steps, endpoint=False)
    t_indices = (t_vals / dt).astype(np.int32)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=t_vals[0], vmax=t_vals[-1])

    fig, ax = plt.subplots()

    for i in t_indices:
        ax.plot(x, Psi[:, i], color=cmap(norm(i * dt)))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\Psi(x, t)$")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, label="$t$")

    plt.tight_layout()
    plt.show()

    return

def animate(Psi_0):
    fig, ax = plt.subplots()

    frame_step = 5

    line = ax.plot(x, Psi_0)[0]
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\Psi(x, t)$")
    ax.set_ylim(-np.max(np.abs(Psi_0)), np.max(np.abs(Psi_0)))

    Psi_now = Psi_0[1:N]
    Psi_prev = Psi_0[1:N]

    # Update function for animation
    def update(frame):
        nonlocal Psi_now, Psi_prev

        for _ in range(frame_step):
            Psi_next = 2 * Psi_now - Psi_prev + c * (dt / dx) ** 2 * D_2.dot(Psi_now)

            Psi_prev = Psi_now
            Psi_now = Psi_next

        Psi_plot = np.insert(Psi_now, (0, N - 1), 0)
        line.set_ydata(Psi_plot)
        ax.set_title("t = " + str(frame * 5 * dt)[:4])

        return 

    ani = animation.FuncAnimation(
        fig, update, interval=10, cache_frame_data=False
    )

    plt.show()

    return

def main():
    # Psi = integrate(initConditions[1], T_steps=150)
    # Psi2 = approx_wave(initConditions[1], time_steps=150,
    #                   mesh_points=x, equations_matrix=D_2,
    #                   dt=dt, c=c, dx=dx, N=N, method='RK')
    #
    # plot(Psi, N_steps=5)
    # plot(Psi2, N_steps=5)

    animate(initConditions[2])

if __name__ == '__main__':
    main()