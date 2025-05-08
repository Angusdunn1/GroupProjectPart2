#Importing modules
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N, L = 256, 30.0#The number of discrete points considered and side length respectively
dx = L / N #incrment between discrete points

x = np.linspace(-L/2, L/2, N, endpoint=False) #x and y arrays defined based on N and L
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij') #arrays explicitely turned into a meshgrid

kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi #defining new meshgrid in fourier space
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
k_squared = KX**2 + KY**2

T = 5 #total simulation run time     
dt = 0.05 #time step increment
steps = int(T / dt) #total number of time steps

##POISSON
def make_source(t):
    x1 = -2 + np.sin(t)
    x2 =  2 - np.sin(t)
    y1 =  2 * np.cos(t)
    y2 = -2 * np.cos(t)
    sigma = 0.5
    u = np.exp(-((X - x1)**2 + (Y - y1)**2) / (2 * sigma**2)) - np.exp(-((X - x2)**2 + (Y - y2)**2) / (2 * sigma**2))
    return u


#Defining axes of plot 
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_title("Poisson Equation with moving sources")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

###POISSON
f0 = make_source(0)
f0_hat = np.fft.fftn(f0)
phi_hat = -f0_hat / k_squared
phi_hat[0, 0] = 0.0
phi = np.real(np.fft.ifftn(phi_hat))
im = ax.imshow(phi, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='seismic', vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Potential [a.u.]")


def update(frame):
    t = frame * dt

    ### POISSON
    f = make_source(t)
    f_hat = np.fft.fftn(f)
    phi_hat = -f_hat / k_squared
    phi_hat[0, 0] = 0.0
    phi = np.real(np.fft.ifftn(phi_hat))
    im.set_array(phi)
    ax.set_title(f"Poisson Solution (t = {t:.2f} s)")

    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=100, repeat= False)

plt.show()