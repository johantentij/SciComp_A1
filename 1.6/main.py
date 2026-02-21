import numpy as np
import matplotlib.pyplot as plt


grid_size = 50
tol = 1e-5
max_iters = 50000

class Grid:
    def __init__(self, N=grid_size):
        self.N = N
        self.resize(N, force=True)

        return

    def resize(self, N, force=False):
        if (self.N == N and not force):
            return

        self.N = N

        X, Y = np.ogrid[:N, :N + 1]
        is_even = (X + Y) % 2 == 0

        self.red_mask = is_even.copy()
        self.black_mask = ~is_even

        self.red_mask[:, 0] = self.red_mask[:, N] = False
        self.black_mask[:, 0] = self.black_mask[:, N] = False

        self.sinks = []
        self.insulators = []

        return
    
    def removeObstacles(self):
        self.resize(self.N, force=True)
    
    def addRectObstacle(self, xy_start, xy_end, obstacleType):
        xs, ys = xy_start
        xe, ye = xy_end

        xs = int(self.N * xs)
        xe = int(self.N * xe)
        ys = int(self.N * ys)
        ye = int(self.N * ye)

        if (obstacleType == 'sink'):
            self.sinks.append((xs, xe, ys, ye))

        elif (obstacleType == 'insulator'):
            self.insulators.append((xs, xe, ys, ye))

        else:
            return False
        
        self.red_mask[xs:xe, ys:ye] = False
        self.black_mask[xs:xe, ys:ye] = False

        return True
    
    def getNeighbours(self, c):
        left    = np.roll(c, -1, axis=0)
        right   = np.roll(c, 1, axis=0)
        up      = np.roll(c, -1, axis=1)
        down    = np.roll(c, 1, axis=1)

        for insulator in self.insulators:
            xs, xe, ys, ye = insulator

            left[xe, ys:ye]     += c[xe, ys:ye]
            right[xs-1, ys:ye]  += c[xs-1, ys:ye]
            up[xs:xe, ye]       += c[xs:xe, ye]
            down[xs:xe, ys-1]   += c[xs:xe, ys-1]

        return left, right, up, down
    
    def getObstaclePlotRects(self):
        dx = 1 / self.N

        x = np.arange(self.N) * dx
        y = np.arange(self.N + 1) * dx

        rects = []
        for insulator in self.insulators:
            xs, xe, ys, ye = insulator
            rects.append(
                plt.Rectangle(
                    (x[xs] - .5 * dx, y[ys] - .5 * dx), 
                    x[xe] - x[xs], 
                    y[ye] - y[ys], 
                    fill=False, 
                    color='white', 
                    hatch='x',
                    ls = ''
                    )
                ) 

        for sink in self.sinks:
            xs, xe, ys, ye = sink
            rects.append(
                plt.Rectangle(
                    (x[xs] - .5 * dx, y[ys] - .5 * dx), 
                    x[xe] - x[xs], 
                    y[ye] - y[ys], 
                    fill=False, 
                    color='red', 
                    hatch='x',
                    ls = ''
                )
            ) 

        return rects
    
    def initC(self):
        c = np.zeros((self.N, self.N + 1))
        c[:, self.N] = 1

        return c

def solve_diffusion(gridObject: Grid, N, omega):
    gridObject.resize(N)
    c = gridObject.initC()
    
    for step in range(1, max_iters + 1):
        c_old = c.copy()
        
        for mask in [gridObject.red_mask, gridObject.black_mask]:
            left, right, up, down = gridObject.getNeighbours(c)
            neighbourSum = left + right + up + down

            c[mask] *= (1 - omega)
            c[mask] += .25 * omega * neighbourSum[mask]
        
        # check for convergence
        max_error = np.max(np.abs(c - c_old))
        if max_error < tol:
            return c, step
            
    return c, max_iters


def question_j(gridMin=10, gridMax=200, gridSteps=20):
    gridObject = Grid(gridMin)

    gridSizes = np.linspace(gridMin, gridMax, gridSteps, dtype=np.int32)
    omegas = np.arange(1.7, 2, 0.005)

    #omega search resolution
    omegaStep = .0005

    bestOmegas = np.empty(gridSteps)
    
    # estimate best omega for gridMin
    iterations = []
    for w in omegas:
        _, iters = solve_diffusion(gridObject, gridMin, w)
        iterations.append(iters)
    
    prevBestOmega = omegas[np.argmin(iterations)]

    for i, gridSize in enumerate(gridSizes):
        # start from optimal value of previous grid size
        w = prevBestOmega
        _, iterations = solve_diffusion(gridObject, gridSizes[i], w)

        # decide left or right
        _, iterations_left = solve_diffusion(gridObject, gridSize, w-omegaStep)
        _, iterations_right = solve_diffusion(gridObject, gridSize, w+omegaStep)

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
                _, iterations = solve_diffusion(gridObject, gridSize, w)

            bestOmegas[i] = w + omegaStep

        else:
            iterations_prev = iterations
            iterations = iterations_right
            # raise omega until minimum is found
            w += omegaStep
            while (iterations <= iterations_prev):
                iterations_prev = iterations
                w += omegaStep
                _, iterations = solve_diffusion(gridObject, gridSize, w)

            bestOmegas[i] = w - omegaStep

        print("Grid size %d, best omega: %f" % (gridSize, bestOmegas[i]))
        prevBestOmega = bestOmegas[i]


    fig, (ax1, ax2) = plt.subplots(1 , 2)
    ax2.plot(gridSizes, bestOmegas)
    ax2.set_xlabel("Grid size (N)")
    ax2.set_ylabel("Optimal $\\omega$")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    # example case with base grid size
    iterations = []
        
    for w in omegas:
        _, iters = solve_diffusion(gridObject, grid_size, omega=w)
        iterations.append(iters)
        print(f"Omega: {w:.2f} | Iterations: {iters}")
        
    optimal_idx = np.argmin(iterations)
    best_omega= omegas[optimal_idx]

    ax1.plot(omegas, iterations, 'ko-', linewidth=2, markersize=3)
    ax1.axvline(best_omega, color='r', linestyle='--', label=f'Optimal: {best_omega:.2f}')
    
    ax1.set_yscale('log')
    ax1.set_title('N = ' + str(grid_size))
    ax1.set_xlabel('Relaxation Factor ($\\omega$)')
    ax1.set_ylabel('No. of iterations until convergence')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    plt.tight_layout()
    plt.show()
    
    print(f"best omega value: Omega={best_omega:.2f}")

    return best_omega

def question_k(optimal_omega):
    gridObject = Grid()

    conditions = ['none', 'sink', 'insulator']
    labels = ['Baseline (No Obstacle)', 'Task K (Sink)', 'Task L (Insulator)']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    dx = 1 / grid_size
    
    x = np.arange(grid_size) * dx
    y = np.arange(grid_size + 1) * dx
    
    for i, cond in enumerate(conditions):
        gridObject.removeObstacles()
        gridObject.addRectObstacle((.2, .4), (.8, .6), cond)

        c, iters = solve_diffusion(gridObject, grid_size, optimal_omega)

        ax = axes[i]
        im = ax.pcolor(x, y, c.T, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{labels[i]}\nConverged in {iters} iters")
        ax.set_aspect("equal")

        rects = gridObject.getObstaclePlotRects()
        for rect in rects:
            ax.add_patch(rect)

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)

    # plt.tight_layout()
    plt.show()

    return

def insulatorMazeTest(N=grid_size):
    gridObject = Grid()

    gridObject.addRectObstacle((.8, .1), (.9, .9), obstacleType='insulator')
    gridObject.addRectObstacle((.1, .1), (.8, .2), obstacleType='insulator')
    gridObject.addRectObstacle((.1, .2), (.2, .9), obstacleType='insulator')
    gridObject.addRectObstacle((.2, .8), (.7, .9), obstacleType='insulator')
    gridObject.addRectObstacle((.6, .3), (.7, .8), obstacleType='insulator')
    gridObject.addRectObstacle((.3, .3), (.6, .4), obstacleType='insulator')
    gridObject.addRectObstacle((.3, .4), (.4, .7), obstacleType='insulator')
    gridObject.addRectObstacle((.4, .6), (.5, .7), obstacleType='insulator')

    # adding a sink in the centre of the maze makes it converge faster
    # gridObject.addRectObstacle((.48, .48), (.52, .52), obstacleType='sink')

    c, iters = solve_diffusion(gridObject, N, 1.91)

    dx = 1 / N
    
    x = np.arange(N) * dx
    y = np.arange(N + 1) * dx

    fig, ax = plt.subplots()
    im = ax.pcolor(x, y, c.T, cmap="inferno", vmin=0, vmax=1)
    rects = gridObject.getObstaclePlotRects()

    for rect in rects:
        ax.add_patch(rect)
    ax.set_title(f"Converged in {iters} iterations")

    plt.colorbar(im, fraction=0.02, pad=0.04)
    plt.show()
    
if __name__ == "__main__":
    # best_omega = question_j()
    # question_k(best_omega)

    insulatorMazeTest()