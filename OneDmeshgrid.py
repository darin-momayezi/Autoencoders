'''Solving dfdt + c*dfdx = 0 on a spatio-temporal grid'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd



##### PARAMETERS #####

T = 8000  # max time
X = 100  # max x
t_list = [100, 1000, 5000, 10000, 20000, 50000]  # Number of temporal grid points
t = int(T)  # Temporal grid points
x = int(X)  # Number of spatial grid points
dt = T / t  # Size of time interval
dx = X / x  # Size of spatial interval

k = 0.001 * 1/dx  # low frequency (k << 1/dx)
q = 1000 * 1/dx  # high frequency (q >> 1/dx)
c = np.pi

a = 1
b = 1



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



##### PLOT SOLUTION #####

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, tt, F, antialiased=False, linewidth=0, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('T')
plt.show()
