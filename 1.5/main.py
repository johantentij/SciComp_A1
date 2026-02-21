import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

N = 50
dx = 1 / N

x = np.arange(N) * dx
y = np.arange(N + 1) * dx

X, Y = np.ogrid[:N, :N + 1]
is_even = (X + Y) % 2 == 0

red_mask = is_even.copy()
black_mask = ~is_even

# Exclude top and bottom fixed boundaries
red_mask[:, 0] = red_mask[:, N] = False
black_mask[:, 0] = black_mask[:, N] = False

def initC():
    c = np.zeros((N, N + 1), dtype=np.float64)
    c[:, N] = 1

    return c

def analyticalSolution():
    return np.outer(np.ones(N), y)

def Jacobi(c, tol=1e-5, returnMaxDiff=False, returnErrors=False):
    maxDiff = 1

    maxDiffHist = []
    errorHist = []
    while maxDiff > tol:
        c_old = np.copy(c)

        c_left = np.roll(c_old, 1, axis=0)
        c_right = np.roll(c_old, -1, axis=0)

        c[:, 1:N] = .25 * (
            c_left[:, 1:N] + 
            c_right[:, 1:N] +
            c_old[:, 0:N-1] + 
            c_old[:, 2:N+1]
        )

        maxDiff = np.max(np.abs(c - c_old))
        maxDiffHist.append(maxDiff)

        #L_2 norm of errors
        errorHist.append(np.sqrt(np.mean((c - analyticalSolution()) ** 2)))

    if returnMaxDiff and returnErrors:
        return c, maxDiffHist, errorHist
    elif returnMaxDiff:
        return c, maxDiffHist
    elif returnErrors:
        return c, errorHist
    else:
        return c

def SOR(c, omega, tol=1e-5, returnMaxDiff=False, returnErrors=False):
    maxDiff = 1

    maxDiffHist = []
    errorHist = []
    while maxDiff > tol:
        c_old = c.copy()
            
        for mask in [red_mask, black_mask]:
            neighbors = (
                np.roll(c, 1, axis=0) + 
                np.roll(c, -1, axis=0) + 
                np.roll(c, 1, axis=1) + 
                np.roll(c, -1, axis=1)
            )

            c[mask] = (1 - omega) * c[mask] + 0.25 * omega * neighbors[mask]
        
        maxDiff = np.max(np.abs(c - c_old))
        maxDiffHist.append(maxDiff)

        #L_2 norm of errors
        errorHist.append(np.sqrt(np.mean((c - analyticalSolution()) ** 2)))

    if returnMaxDiff and returnErrors:
        return c, maxDiffHist, errorHist
    elif returnMaxDiff:
        return c, maxDiffHist
    elif returnErrors:
        return c, errorHist
    else:
        return c

def Gauss_Seidel(c, tol=1e-5):
    # Gauss-Seidel as special case of SOR:
    return SOR(c, omega=1, tol=tol)

def questionH():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    _, errors_Jacobi = Jacobi(initC(), returnErrors=True)
    ax1.plot(errors_Jacobi)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_xlabel("No. of iteration")
    ax1.set_ylabel("$L_2$ norm of errors")
    ax1.set_title("Jacobi")

    omegas = [.5, 1, 1.5, 1.8, 1.9]
    for omega in omegas:
        _, errors_SOR = SOR(initC(), omega, returnErrors=True)

        if (omega == 1):
            ax2.plot(errors_SOR, label=f"$\\omega = ${omega:.2f} (Gauss-Seidel)")

        else:
            ax2.plot(errors_SOR, label=f"$\\omega = ${omega:.2f}")

    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_xlabel("No. of iteration")
    ax2.set_ylabel("$L_2$ norm of errors")
    ax2.set_title("SOR and Gauss-Seidel")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return

def question_h(N=50):
    omega = 1.8
    
    c_jac = Jacobi(initC())
    c_gs = SOR(initC(), omega=1)
    c_sor = SOR(initC(), omega)
    
    y_vals = np.linspace(0, 1, N + 1)

    fig, ax = plt.subplots()
    
    # left plot
    step = 2 

    # right plot: residuals
    res_jac = np.abs(c_jac[0, :] - y_vals)
    res_gs = np.abs(c_gs[0, :] - y_vals)
    res_sor = np.abs(c_sor[0, :] - y_vals)
    
    ax.semilogy(y_vals, res_jac, 'o-', markersize=4, label='Jacobi', alpha=0.7)
    ax.semilogy(y_vals, res_gs, 's-', markersize=4, label='Gauss-Seidel', alpha=0.7)
    ax.semilogy(y_vals, res_sor, '^-', markersize=4, label=f'SOR ($\omega={omega}$)', alpha=0.7)
    
    ax.set_title(f"$|c_{{num}} - c_{{ana}}|$ (N={N})")
    ax.set_xlabel("y coordinate")
    ax.set_ylabel("Absolute Error (Log Scale)")
    ax.legend()
    ax.grid(True, which='both', ls=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

def questionI():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    _, maxDiffHist_Jacobi = Jacobi(initC(), returnMaxDiff=True)
    ax1.plot(maxDiffHist_Jacobi)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_xlabel("No. of iteration")
    ax1.set_ylabel("$\\delta$")
    ax1.set_title("Jacobi")

    omegas = [.5, 1, 1.5, 1.8, 1.9]
    for omega in omegas:
        _, maxDiffHist_SOR = SOR(initC(), omega, returnMaxDiff=True)

        if (omega == 1):
            ax2.plot(maxDiffHist_SOR, label=f"$\\omega = ${omega:.2f} (Gauss-Seidel)")

        else:
            ax2.plot(maxDiffHist_SOR, label=f"$\\omega = ${omega:.2f}")

    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_xlabel("No. of iteration")
    ax2.set_ylabel("$\\delta$")
    ax2.set_title("SOR and Gauss-Seidel")
    ax2.legend()

    # plt.tight_layout()
    plt.show()

    return

# questionH()
question_h()





