import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def __step(psi, y, equations_matrix, dx, dt, c, time_steps)

D  = 1.0       # diffusion constant
N  = 50        # number of intervals (grid: (N+1) x (N+1))
dx = 1.0 / N   # spatial step
dt = 0.9 * dx**2 / (4 * D)   # 90% of stability limit
t_end      = 1.0
time_steps = int(t_end / dt)
c = np.zeros((N, N+1, time_steps + 1))
c[:, N, :] = 1.0   # top:    c(x, y=1, t) = 1
c[:, 0, :] = 0.0   # bottom: c(x, y=0, t) = 0
mask = np.ones((N, N+1), dtype=bool)


def solve_steady_state_3d(N=50, omega=1.7, N_iter=2000, tol=10e-5, save_every=5, sink_slice=None, insul_slice=None):
    """
    Returns:
        c_3d (np.ndarray): 3D array of shape (Saved_Steps, N, N+1)
        steps (list): The specific iteration numbers saved in c_3d
    """
    # 1. INITIALIZE THE 2D WORKING GRID
    # N points for periodic x, N+1 points for fixed y
    c_current = np.zeros((N, N + 1))

    # Apply Boundary Conditions
    c_current[:, N] = 1.0  # Top: c(x, y=1, t) = 1
    c_current[:, 0] = 0.0  # Bottom: c(x, y=0, t) = 0

    # Define Sink
    if sink_slice:
        c_current[sink_slice] = 0

    # 2. PREPARE STORAGE
    history = [c_current.copy()]
    steps = [0] # Tracking iteration counts

    # 3. SET UP RED-BLACK MASKS
    X, Y = np.ogrid[:N, :N + 1]
    red_mask = (X + Y) % 2 == 0
    black_mask = (X + Y) % 2 == 1

    # Protect boundaries and sink from updates
    for mask in [red_mask, black_mask]:
        mask[:, 0] = False
        mask[:, N] = False
        if sink_slice:
            mask[sink_slice] = False
        if insul_slice:
            mask[insul_slice] = False

    # 4. RUN SOR ITERATIONS
    for i in range(1, N_iter + 1):
        c_old = c_current.copy()

        # --- UPDATE RED CELLS ---
        c_left = np.roll(c_current, 1, axis=0)
        c_right = np.roll(c_current, -1, axis=0)
        c_down = np.roll(c_current, 1, axis=1)
        c_up = np.roll(c_current, -1, axis=1)

        c_current[red_mask] = (1 - omega) * c_current[red_mask] + \
                              0.25 * omega * (c_left[red_mask] + c_right[red_mask] + c_up[red_mask] + c_down[red_mask])

        # --- UPDATE BLACK CELLS ---
        c_left = np.roll(c_current, 1, axis=0)
        c_right = np.roll(c_current, -1, axis=0)
        c_down = np.roll(c_current, 1, axis=1)
        c_up = np.roll(c_current, -1, axis=1)

        c_current[black_mask] = (1 - omega) * c_current[black_mask] + \
                                0.25 * omega * (c_left[black_mask] + c_right[black_mask] + c_up[black_mask] + c_down[
            black_mask])

        # Ensure sink stays at 0.0
        if sink_slice:
            c_current[sink_slice] = 0.0

        # if insul_slice:
        #     # For a simple rectangular block, we can set the internal values
        #     # to the average of the neighbors outside to simulate the 'no-flow' boundary
        #     c_current[insul_slice] = 0.25 * (c_left[insul_slice] + c_right[insul_slice] +
        #                                      c_up[insul_slice] + c_down[insul_slice])

        if insul_slice:
            xs, xe = insul_slice[0].start, insul_slice[0].stop
            ys, ye = insul_slice[1].start, insul_slice[1].stop

            # Left face
            c_current[xs, ys:ye] = c_current[xs-1, ys:ye]

            # Right face
            c_current[xe-1, ys:ye] = c_current[xe, ys:ye]

            # Bottom face
            c_current[xs:xe, ys] = c_current[xs:xe, ys-1]

            # Top face
            c_current[xs:xe, ye-1] = c_current[xs:xe, ye]


        # 5. SAVE PERIODICALLY
        if i % save_every == 0:
            history.append(c_current.copy())
            steps.append(i)

        # 6. CONVERGENCE CHECK
        if np.max(np.abs(c_current - c_old)) < tol:
            if steps[-1] != i:  # Don't duplicate if already saved by save_every
                history.append(c_current.copy())
                steps.append(i)
            print(f"Converged after {i} iterations.")
            break

    return history, steps


# Example usage:
c_3d, saved_iterations = solve_steady_state_3d(insul_slice=(slice(20,25),slice(20,25)))

time_steps = len(saved_iterations)

# --- Part F: Snapshots of the Relaxation Process ---
# We'll pick a few indices from our 'saved_iterations' list to show progress
num_snaps = 5
# Select indices evenly spaced through the history
snap_indices = np.linspace(0, len(saved_iterations) - 1, num_snaps, dtype=int)

fig_f, axes = plt.subplots(1, num_snaps, figsize=(16, 3.5))

# Generate X, Y mesh matching your (N, N+1) grid
x_coords = np.linspace(0, 1, N)
y_coords = np.linspace(0, 1, N + 1)
X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')

for ax, idx in zip(axes, snap_indices):
    # c_3d has shape (Iteration, X, Y)
    im = ax.pcolormesh(X_mesh, Y_mesh, c_3d[idx], cmap="inferno",
                       vmin=0, vmax=1, shading="auto")

    iter_val = saved_iterations[idx]
    ax.set_title(f"Iteration {iter_val}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig_f.suptitle("SOR Steady-State Convergence (with Sink)", fontsize=13)
fig_f.tight_layout()
plt.savefig("norm_convergence.png")
plt.show()

# --- Part G: Convergence Animation ---
fig_g, ax_g = plt.subplots(figsize=(5, 5))

# Initial frame
im_g = ax_g.pcolormesh(X_mesh, Y_mesh, c_3d[0], cmap="inferno",
                       vmin=0, vmax=1, shading="auto")
plt.colorbar(im_g, ax=ax_g)
ax_g.set_xlabel("x")
ax_g.set_ylabel("y")
ax_g.set_aspect("equal")
title_g = ax_g.set_title(f"Iteration: {saved_iterations[0]}")


def update(frame):
    # Update the mesh data with the next 2D slice
    # pcolormesh set_array expects a 1D array of the face values
    im_g.set_array(c_3d[frame].ravel())
    title_g.set_text(f"Iteration: {saved_iterations[frame]}")
    return [im_g, title_g]


ani = animation.FuncAnimation(fig_g, update, frames=len(saved_iterations),
                              interval=50, blit=True)

# Save the animation
ani.save("norm_sor_convergence.gif", fps=20, writer='pillow')
print("Saved sor_convergence.gif")

# plt.show()
print("Done.")