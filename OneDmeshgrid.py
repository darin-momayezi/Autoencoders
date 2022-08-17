'''Solving dfdt + c*dfdx = 0 on a spatio-temporal grid'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd


use_Fst = True
length = 3


##### PARAMETERS #####

T = 10000  # temporal domain
X = 100  # spatial domain
x = 5  # spatial grid points
t = 1000  # temporal grid points
dx = X / x  # spatial step size
dt = T / t  # temporal step size

k = 0.01 * 1/dx  # low frequency (k << 1/dx)
q = 100 * 1/dx  # high frequency (q >> 1/dx)
c = np.pi

a = 1  # low frequency on/off
b = 1  # high frequency on/off



##### DATA #####

x_arr = np.linspace(0, X, x)  # Spatial data

t_arr = np.linspace(0, T, t)  # Temporal data\
    
xx, tt = np.meshgrid(x_arr, t_arr)  # Spatio-temporal grid

xx = np.round(xx, 0)  # Integer spatial grid points
tt = np.round(tt, 0)  # Integer temporal grid points


u = xx + c*tt

F = a*np.cos(k*u) + b*np.cos(q*u)  # Solution

Fst = []

if use_Fst == True:
    for idx, row in enumerate(F):  # Reshape F into spatio-temporal vectors (Fst)
        
        if (idx+1) % 3 == 0:
            Fst.append(np.concatenate((F[idx-2], F[idx-1], F[idx])))
        else: continue
        
    Fst = np.array(Fst)
print(Fst)


##### SAVE DATA #####
if use_Fst == True:
    np.save('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.npy', Fst)
    print(Fst.shape)
elif use_Fst == False:
    np.save('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.npy', F)
    print(F.shape)


pd.DataFrame(F.T).to_csv('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.csv', header=None, index=None)



##### PLOT SOLUTION #####

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, tt, F, antialiased=False, linewidth=0, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('T')
plt.show()
