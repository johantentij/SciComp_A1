import numpy as np
import matplotlib.pyplot as plt
import time

def jacobi(N=50, tolerance=1e-5, max_iter=10000):
    grid = np.zeros((N, N))
    grid_new = np.zeros((N, N))
    
    grid[0, :] = 1  # top row
    grid[-1, :] = 0  # bottom row
    
    start_time = time.time()
    
    for k in range(max_iter):
        grid_old = grid.copy()
        
        # iterate by rows
        for i in range(1, N-1):
            
            # iterate by column
            for j in range(N):
                up = grid_old[i-1, j]
                down = grid_old[i+1, j]
                
                if j == 0:
                    # left boundary
                    left = grid_old[i, N-1]
                else:
                    left = grid_old[i, j-1]
                    
                if j == N-1:
                    # right boundary
                    right = grid_old[i, 0]
                else:
                    right = grid_old[i, j+1]
                
                grid_new[i, j] = (up + down + left + right) / 4
        
        # reset boundary
        grid_new[0, :] = 1
        grid_new[-1, :] = 0
        
        # check tolerance
        diff = np.abs(grid_new - grid_old).max()
        grid = grid_new
        
        if diff < tolerance:
            print(f"converged with {k} iterations with a difference of {diff}")
            break
            
    print(f"time spent: {time.time() - start_time:.2f} seconds")
    return grid


result_grid = jacobi(N=50)

# visualization
plt.figure(figsize=(6, 5))
plt.imshow(result_grid, cmap='gray_r') 

plt.colorbar(label='Concentration')
plt.title('Jacobi Iteration Diffusion')
plt.show()