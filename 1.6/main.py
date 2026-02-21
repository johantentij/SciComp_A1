import numpy as np
import matplotlib.pyplot as plt


grid_size = 50
tol = 1e-5
max_iters = 50000

def initialize_grid(N):
    c = np.zeros((N, N + 1))
    c[:, N] = 1
    return c

def get_red_black_masks(N, obs_bounds, obs_type):
    """
    Generate boolean masks for Red-Black parallelized updates.
    Excludes boundaries and internal obstacle regions to prevent overwriting.
    """
    X, Y = np.ogrid[:N, :N + 1]
    is_even = (X + Y) % 2 == 0
    
    red_mask = is_even.copy()
    black_mask = ~is_even
    
    # Exclude top and bottom fixed boundaries
    red_mask[:, 0] = red_mask[:, N] = False
    black_mask[:, 0] = black_mask[:, N] = False
    
    # Exclude internal obstacle region from standard SOR updates
    if obs_type in ['sink', 'insulator']:
        xs, xe, ys, ye = obs_bounds
        red_mask[xs:xe, ys:ye] = False
        black_mask[xs:xe, ys:ye] = False
        
    return red_mask, black_mask

def sim_obstacle(c, obs_bounds, obs_type):

    if obs_type == 'none':
        return
        
    xs, xe, ys, ye = obs_bounds
    
    if obs_type == 'sink':
        # sink absorbs all heat and concentration drops to 0
        c[xs:xe, ys:ye] = 0.0
        
    elif obs_type == 'insulator':
        # set the edge of the insulator the same as adjacent fluid node
        c[xs, ys:ye] = c[xs-1, ys:ye]

        # Right boundary
        c[xe-1, ys:ye] = c[xe, ys:ye]

        # Bottom boundary
        c[xs:xe, ys] = c[xs:xe, ys-1]

        # Top boundary
        c[xs:xe, ye-1] = c[xs:xe, ye]


def solve_diffusion(N, omega, obs_type='none'):
    c = initialize_grid(N)
    
    # define obstacle size and place
    obs_bounds = (int(N * 0.4), int(N * 0.6), int((N+1) * 0.4), int((N+1) * 0.6))
    red_mask, black_mask = get_red_black_masks(N, obs_bounds, obs_type)
    
    for step in range(1, max_iters + 1):
        sim_obstacle(c, obs_bounds, obs_type)
        c_old = c.copy()
        
    

        # Vectorized updates using NumPy roll for parallelized neighbor access
        for mask in [red_mask, black_mask]:
            neighbors = (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) + 
                         np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1))
            c[mask] = (1 - omega) * c[mask] + 0.25 * omega * neighbors[mask]
            
        
        
        # check for convergence
        max_error = np.max(np.abs(c - c_old))
        if max_error < tol:
            return c, step
            
    return c, max_iters


def question_j(gridMin=10, gridMax=200, gridSteps=20):
    gridSizes = np.linspace(gridMin, gridMax, gridSteps, dtype=np.int32)
    omegas = np.arange(1, 2.01, 0.05)

    #omega search resolution
    omegaStep = .0005

    bestOmegas = np.empty(gridSteps)
    
    # estimate best omega for gridMin
    iterations = []
    for w in omegas:
        _, iters = solve_diffusion(gridMin, w)
        iterations.append(iters)
    
    prevBestOmega = omegas[np.argmin(iterations)]

    for i, gridSize in enumerate(gridSizes):
        # start from optimal value of previous grid size
        w = prevBestOmega
        _, iterations = solve_diffusion(gridSizes[i], w)

        # decide left or right
        _, iterations_left = solve_diffusion(gridSize, w-omegaStep)
        _, iterations_right = solve_diffusion(gridSize, w+omegaStep)

        if (iterations_right > iterations and iterations_left > iterations):
            # starting value was optimal
            bestOmegas[i] = w

        elif (iterations_left < iterations_right):
            iterations_prev = iterations
            iterations = iterations_left
            # lower omega until minimum is found
            w -= omegaStep
            while (iterations <= iterations_prev):
                iterations_prev = iterations
                w -= omegaStep
                _, iterations = solve_diffusion(gridSize, w)

            bestOmegas[i] = w + omegaStep

        else:
            iterations_prev = iterations
            iterations = iterations_right
            # raise omega until minimum is found
            w += omegaStep
            while (iterations <= iterations_prev):
                iterations_prev = iterations
                w += omegaStep
                _, iterations = solve_diffusion(gridSize, w)

            bestOmegas[i] = w - omegaStep

        print("Grid size %d, best omega: %f" % (gridSize, bestOmegas[i]))
        prevBestOmega = bestOmegas[i]


    plt.figure(figsize=(8, 5))
    plt.plot(gridSizes, bestOmegas)
    plt.xlabel("grid size")
    plt.ylabel("Optimal $\\omega$")
    
    # example case with base grid size
    iterations = []
        
    for w in omegas:
        _, iters = solve_diffusion(grid_size, omega=w, obs_type='none')
        iterations.append(iters)
        print(f"Omega: {w:.2f} | Iterations: {iters}")
        
    optimal_idx = np.argmin(iterations)
    best_omega= omegas[optimal_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, iterations, 'ko-', linewidth=2, markersize=6)
    plt.axvline(best_omega, color='r', linestyle='--', label=f'Optimal: {best_omega:.2f}')
    
    plt.yscale('log')
    plt.title('N = ' + str(grid_size))
    plt.xlabel('Relaxation Factor ($\omega$)')
    plt.ylabel('Iterations to Converge (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"best omega value: Omega={best_omega:.2f}")

    return best_omega

def question_k(optimal_omega):
    
    conditions = ['none', 'sink', 'insulator']
    labels = ['Baseline (No Obstacle)', 'Task K (Sink)', 'Task L (Insulator)']
    results = []
    
    for cond in conditions:
        c, iters = solve_diffusion(grid_size, optimal_omega, obs_type=cond)
        results.append((c, iters))
        
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    X, Y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size+1), indexing='ij')
    
    for i, ax in enumerate(axes):
        c_matrix, iters = results[i]
        im = ax.pcolor(X, Y, c_matrix, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{labels[i]}\nConverged in {iters} iters")
        ax.set_aspect("equal")
        if i > 0:
            rect = plt.Rectangle((0.4, 0.4), 0.2, 0.2, fill=False, color='white', ls='--')
            ax.add_patch(rect)

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.show()
    
if __name__ == "__main__":
    best_omega = question_j()
    # question_k(1.8)