'''Solving dfdt + c*dfdx = 0 on a spatio-temporal grid'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd



##### PARAMETERS #####

T = 10000  # max time
X = 200  # max x
t = int(T)  # Temporal grid points
x = int(X)  # Number of spatial grid points
dt = T / t  # Size of time interval
dx = X / x  # Size of spatial interval

k = 0.01 * 1/dx  # low frequency (k << 1/dx)
q = 100 * 1/dx  # high frequency (q >> 1/dx)
c = np.pi

a = 1
b = 1



##### DATA #####

x_arr = np.linspace(0, X, x)  # Spatial data

t_arr = np.linspace(0, T, t)  # Temporal data

xx, tt = np.meshgrid(x_arr, t_arr)  # Spatio-temporal grid

xx = np.round(xx, 0)  # Integer spatial grid points
tt = np.round(tt, 0)  # Integer temporal grid points

u = xx + c*tt

F = a*np.cos(k*u) + b*np.cos(q*u)  # Solution

Fst = np.ones((int(F.shape[0]/2), int(2*F.shape[1])))

for idx, row in enumerate(F):  # Reshape F into spatio-temporal vectors (Fst)

    if idx % 2 == 0 and idx != 0:
        Fst[int(idx/2)] = np.concatenate([F[idx-1], F[idx]])
    else: continue



##### SAVE DATA #####

np.save('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.npy', Fst)

pd.DataFrame(F.T).to_csv('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.csv', header=None, index=None)



##### PLOT SOLUTION #####

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, tt, F, antialiased=False, linewidth=0, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('T')
plt.show()
