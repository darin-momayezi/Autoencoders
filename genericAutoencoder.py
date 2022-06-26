'''Solving dfdt + c*dfdx = 0 on a spatio-temporal grid'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fft2, ifft2, fftn, ifftn, fftfreq



##### PARAMETERS #####

T = 50000  # max time
X = 1000  # max x
t_list = [100, 1000, 5000, 10000, 20000, 50000]  # Number of temporal grid points
t = t_list[2]  # Temporal grid points
x = int(X/10)  # Number of spatial grid points
dt = T / t  # Size of time interval
dx = X / x  # Size of spatial interval

k = 0.001 * 1/dx  # low frequency (k << 1/dx)
q = 1000 * 1/dx  # high frequency (q >> 1/dx)
c = 10*np.pi

a = 1
b = 0



##### DATA #####

x_arr = np.linspace(0, X, x)  # Spatial data

t_arr = np.linspace(0, T, t)  # Temporal data

xx, tt = np.meshgrid(x_arr, t_arr)
xx = np.round(xx, 0)
tt = np.round(tt, 0)


u = xx + c*tt

F = a*np.cos(k*u) + b*np.cos(q*u)

##### SAVE DATA #####

np.save('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dcoarsegrid.npy', F)

pd.DataFrame(F.T).to_csv('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dcoarsegrid.csv', header=None, index=None)



# Fourier transform
# fourier = fft2(F)
# freq = fftfreq(x, dt)
# for row in fourier:
#     for idx, x in enumerate(row):
#         if freq[idx] == 0:
#             continue
#         else:
#             x = x / freq[idx]

# F = ifft2(fourier)




# Finite difference
# F = np.zeros((n, m))  # Solution
# for idt, t in enumerate(t_arr):
    
#     for idx, x in enumerate(x_arr):
        
#         if idx != 0:
#             F[idx, idt] = F[idx-1, idt-1] * ((1/dt + c/dx) / (-1/dt - c/dx))
#         else:
#             continue

# Save solution
# np.savetxt('/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/1D_PDEs/1Dcoarsegrid.csv', fourier, delimiter=',')



##### PLOT SOLUTION #####

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, tt, F, antialiased=False, linewidth=0, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('T')
plt.show()

