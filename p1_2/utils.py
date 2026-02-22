import numpy as np
from p1_1.euler_methods import approx_heat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc

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

def analytical(y, t, D=1.0, n_terms=200):
    if t == 0:
        return np.zeros_like(y, dtype=float)
    c = np.zeros_like(y, dtype=float)
    sqrt2Dt = np.sqrt(2 * D * t)
    for i in range(n_terms):
        c += erfc((1 - y + 2*i) / sqrt2Dt) - erfc((1 + y + 2*i) / sqrt2Dt)
    return c

# Snapshot helper: index directly into 3D matrix
def get_snap(t):
    k = min(int(round(t / dt)), time_steps)
    return c[:, :, k]

snap_times = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
snaps = {t: get_snap(t) for t in snap_times}

y_vals = np.linspace(0, 1, N+1)
x_vals = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")


c = approx_heat(c_0=c,dx=dx,time_steps=time_steps, dt=dt, D=D, N=N, method="EF")


# Part E:
fig_e, ax_e = plt.subplots(figsize=(7, 5))
colours = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_times)))

for col, t_snap in zip(colours, snap_times):
    c_num = snaps[t_snap].mean(axis=0)
    c_ana = analytical(y_vals, t_snap) if t_snap > 0 else np.zeros(N+1)
    label = f"t = {t_snap}"
    ax_e.plot(y_vals, c_num, color=col, lw=2,        label=f"Num  {label}")
    ax_e.plot(y_vals, c_ana, color=col, lw=1.5, ls="--", label=f"Ana  {label}")

ax_e.set_xlabel("y"); ax_e.set_ylabel("c(y)")
ax_e.set_title("Part E - Numerical vs Analytical solution")
ax_e.legend(fontsize=7, ncol=2); ax_e.grid(alpha=0.3)
fig_e.tight_layout()
fig_e.savefig("heat_diffusion.png")

# Part F
fig_f, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 12))
axes_flat = axes.flatten()
for i, t_snap in enumerate(snap_times):
    ax = axes_flat[i]
    im = ax.pcolormesh(X, Y, snaps[t_snap], cmap="inferno",
                       vmin=0, vmax=1, shading="auto")
    ax.set_title(f"t = {t_snap}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(len(snap_times), len(axes_flat)):
    axes_flat[j].axis('off')

fig_f.suptitle("Part F - 2D concentration field", fontsize=13)
fig_f.tight_layout()
fig_f.savefig("partF_2Dplots.png", dpi=150)
print("Saved partF_2Dplots.png")

# Part G
n_frames = 120
frame_indices = np.linspace(0, time_steps, n_frames, dtype=int)

fig_g, ax_g = plt.subplots(figsize=(5, 5))
im_g = ax_g.pcolormesh(X, Y, c[:, :, 0], cmap="inferno",
                        vmin=0, vmax=1, shading="auto")
plt.colorbar(im_g, ax=ax_g)
ax_g.set_xlabel("x"); ax_g.set_ylabel("y"); ax_g.set_aspect("equal")
title_g = ax_g.set_title("t = 0.0000")

def update(frame):
    k = frame_indices[frame]
    im_g.set_array(c[:, :, k].ravel())
    title_g.set_text(f"t = {k * dt:.4f}")
    return [im_g, title_g]

ani = animation.FuncAnimation(fig_g, update, frames=n_frames,
                               interval=50, blit=True)
ani.save("partG_animation.gif", fps=20)
print("Saved partG_animation.gif")