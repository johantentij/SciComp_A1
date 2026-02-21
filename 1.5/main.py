import numpy as np
import matplotlib.pyplot as plt

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

    plt.tight_layout()
    plt.show()

    return

questionH()





