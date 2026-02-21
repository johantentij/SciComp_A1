from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def plot(Psi, x, N_steps=10, dt=0.01):
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

def animate_wave(psi, x, dt, frame_skip=5, save_path=None):
    """
    Animate the wave solution.

    Parameters:
    -----------
    psi : ndarray, shape (N+1, time_steps)
        Solution array from approx_wave
    x : ndarray, shape (N+1,)
        Spatial mesh points
    dt : float, int
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

def largerPlotFont():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    return