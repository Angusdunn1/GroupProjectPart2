#Importing modules
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N, L = 256, 10.0#The number of discrete points considered and side length respectively
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

###DIFFUSION
alpha = 0.2 #diffusion coefficient 
sigma = 1
u0 = np.exp(-((X - 2)**2 + (Y - 2)**2) / (2 * sigma**2)) + np.exp(-((X + 2)**2 + (Y + 2)**2) / (2 * sigma**2)) + np.exp(-((X - 2)**2 + (Y + 2)**2) / (2 * sigma**2)) + np.exp(-((X + 2)**2 + (Y - 2)**2) / (2 * sigma**2)) #two gaussian hotspots with width sigma for inital conditions
u_hat = np.fft.fftn(u0) #inital conditions transformed into fouroer space
time_operator = np.exp(-alpha * k_squared * dt) #Fourier transformed diffusion equation (will have to explain fourier transform in report)

#Defining axes of plot 
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_title("2D Heat Diffusion")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

###HEAT DIFFUSION
u = np.real(np.fft.ifftn(u_hat)) #taking the real part of the inverse of the fourier transform of the initial conditions
im = ax.imshow(u, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='hot', vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Temperature [a.u.]")

def update(frame):
    t = frame * dt

    ### DIFFUSION
    global u_hat
    u_hat *= time_operator
    u = np.real(np.fft.ifftn(u_hat))
    im.set_array(u)
    ax.set_title(f"2D Heat Diffusion (t = {t:.2f} s)")

    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=100, repeat= False)

plt.show()