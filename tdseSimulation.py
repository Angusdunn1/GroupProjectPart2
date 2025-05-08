#Importing modules
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

# ###SCHROEDINGER
hbar = 1.0
m = 1.0
sigma = 0.5
k0x, k0y = 4.0, 4.0
u0 = np.exp(-((X)**2 + (Y)**2) / (2 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y))# Gaussian wave packet, modulated by a plane wave (gives it momentum)
u_hat = np.fft.fftn(u0) #initial wavefunction in Fourier space
time_operator = np.exp(-1j * hbar * k_squared * dt / (2 * m)) #Unitary evolution operator for the Schrödinger equation in Fourier space

#Defining axes of plot 
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_title("2D Free-Particle Schrödinger Evolution")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")


###SCHROEDINGER
u = np.fft.ifftn(u_hat)
density = np.abs(u)**2
im = ax.imshow(density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='viridis', vmin=0, vmax=0.1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Probability Density [a.u.]")

def update(frame):
    t = frame * dt

    ### SCHROEDINGER
    global u_hat
    u_hat *= time_operator
    u = np.fft.ifftn(u_hat)
    density = np.abs(u)**2
    im.set_array(density)
    ax.set_title(f"2D Free-Particle, Schrödinger Evolution (t = {t:.2f} s)")

    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=100, repeat=False)

# Save the animation as a GIF
ani.save('schrodinger_evolution.gif', writer='pillow', fps=20)
plt.close()