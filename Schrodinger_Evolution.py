#!/usr/bin/env python3
"""
2D Free-Particle Schrödinger Evolution Simulation
This script simulates and visualizes the time evolution of a quantum wave packet
in two dimensions, saving the animation as a GIF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_schrodinger_evolution(
    N=512,  # Number of grid points
    L=30.0,  # Domain size
    T=5.0,  # Total simulation time
    dt=0.05,  # Time step
    sigma=0.5,  # Initial wave packet width
    k0x=1.0,  # Initial x-momentum
    k0y=0.0,  # Initial y-momentum
    hbar=1.0,  # Reduced Planck constant
    m=1.0,  # Particle mass
    output_file='schrodinger_evolution.gif'  # Output GIF filename
):
    """Run the Schrödinger evolution simulation and save as GIF."""
    # Grid setup
    dx = L / N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    y = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Fourier space setup
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_squared = KX**2 + KY**2

    # Initial conditions
    u0 = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y))
    u_hat = np.fft.fftn(u0)
    time_operator = np.exp(-1j * hbar * k_squared * dt / (2 * m))

    # Total steps
    steps = int(T / dt)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Initial plot
    u = np.fft.ifftn(u_hat)
    density = np.abs(u)**2
    im = ax.imshow(
        density,
        extent=[-L/2, L/2, -L/2, L/2],
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=0.1
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability Density [a.u.]")

    def update(frame):
        """Update function for animation."""
        nonlocal u_hat
        
        # Time evolution in Fourier space
        u_hat *= time_operator
        
        # Transform back to real space
        u = np.fft.ifftn(u_hat)
        density = np.abs(u)**2
        
        # Update plot
        im.set_array(density)
        ax.set_title(f"2D Free-Particle Schrödinger Evolution (t = {frame * dt:.2f} s)")
        
        return [im]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        blit=True
    )

    # Save animation as GIF
    anim.save(output_file, writer='pillow', fps=20)
    plt.close()


if __name__ == '__main__':
    run_schrodinger_evolution(
        T=5.0,  # 5 seconds of simulation time
        output_file='schrodinger_evolution.gif'
    )