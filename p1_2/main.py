import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc

def analytical(y, t, D=1.0, n_terms=200):
    if t == 0:
        return np.zeros_like(y, dtype=float)
    c = np.zeros_like(y, dtype=float)
    sqrt2Dt = np.sqrt(2 * D * t)
    for i in range(n_terms):
        c += erfc((1 - y + 2*i) / sqrt2Dt) - erfc((1 + y + 2*i) / sqrt2Dt)
    return c

D  = 1.0       # diffusion constant
N  = 50        # number of intervals (grid: (N+1) x (N+1))
dx = 1.0 / N   # spatial step
dt = 0.9 * dx**2 / (4 * D)   # 90% of stability limit
t_end      = 1.0
time_steps = int(t_end / dt)

alpha = dt * D / dx**2  

c = np.zeros((N, N+1, time_steps + 1))
c[:, N, 0] = 1


for k in range(time_steps):
    c[:, N, k+1] = 1
    c[:, 0, k+1] = 0
    
    for i in range(N):
        left_i  = (i - 1) % N
        right_i = (i + 1) % N
        
        for j in range(1, N):
            up_j   = j + 1
            down_j = j - 1
            
            c[i, j, k+1] = c[i, j, k] + alpha * (
                c[right_i, j, k] + 
                c[left_i,  j, k] + 
                c[i, up_j,   k] + 
                c[i, down_j, k] - 
                4 * c[i, j, k]
            )