#!/usr/bin/env python3
"""
Example script that runs the disease spread simulation and saves it as a GIF.
This demonstrates how to use the simulation module and create an animated visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import map_coordinates


def run_simulation_and_save_gif(
    original_N=200,  # Initial number of agents per unit area
    domain_size=2.0,  # Size of the square domain
    grid_size=64,  # Resolution of the infection field
    D=0.1,  # Diffusion coefficient
    dt=0.01,  # Time step
    total_time=1000.0,  # Total simulation time
    move_scale=0.0001,  # Step length per dt
    plot_interval=50,  # Steps between plot updates
    recovery_time=30.0,  # Mean infection duration
    recovery_spread=0.2,  # ±20% variability in recovery time
    immunity_multiplier=1.0,  # Immune duration = multiplier × recovery time
    station_rel=(0.9, 0.1),  # Vaccination station position (relative)
    station_radius=0.1,  # Vaccination station radius
    vacc_prob=0.05,  # Vaccination probability per visit
    beta=0.005,  # Mean transmissibility
    beta_spread=0.2,  # ±20% variability in transmissibility
    min_seg_steps=50,  # Min length of straight segment
    max_seg_steps=600,  # Max length of straight segment
    output_file='simulation.gif'  # Output GIF filename
):
    """Run the simulation and save it as a GIF animation."""
    # Total iterations and agent count
    steps = int(total_time / dt)
    N_agents = int(original_N * domain_size**2)

    # Agent-specific transmissibility and recovery
    beta_i = np.random.uniform(
        (1-beta_spread)*beta,
        (1+beta_spread)*beta,
        size=N_agents
    )
    tau_i = np.random.uniform(
        (1-recovery_spread)*recovery_time,
        (1+recovery_spread)*recovery_time,
        size=N_agents
    )

    # Initial positions and motion state
    pos = np.random.rand(N_agents, 2) * domain_size
    phi = 2*np.pi * np.random.rand(N_agents)
    seg_steps = np.random.randint(
        min_seg_steps,
        max_seg_steps+1,
        size=N_agents
    )

    # Infection / vaccination state arrays
    infected = np.zeros(N_agents, bool)
    infected[0] = True  # Start with one infected agent
    vaccinated = np.zeros(N_agents, bool)
    immune = np.zeros(N_agents, bool)
    t_inf = np.zeros(N_agents, float)
    t_immune_max = np.zeros(N_agents, float)
    t_immune_age = np.zeros(N_agents, float)

    # Vaccination station setup
    station_center = np.array(station_rel) * domain_size

    # Spectral diffusion setup
    dx = domain_size / grid_size
    k = np.fft.fftfreq(grid_size, d=dx) * 2*np.pi
    k2 = -(k[:,None]**2 + k[None,:]**2)
    diff_factor = np.exp(D * k2 * dt)

    # Set up figure for animation
    fig, (ax_scat, ax_line) = plt.subplots(2, 1, figsize=(6, 9))
    
    # Scatter plot setup
    scat = ax_scat.scatter(pos[:,0], pos[:,1], s=30)
    station = plt.Circle(
        station_center, 
        station_radius,
        edgecolor='grey', 
        facecolor='none',
        lw=2, 
        alpha=0.7
    )
    ax_scat.add_patch(station)
    ax_scat.set_xlim(0, domain_size)
    ax_scat.set_ylim(0, domain_size)
    ax_scat.set_aspect('equal', adjustable='box')
    ax_scat.set_xticks([])
    ax_scat.set_yticks([])

    # Line plot setup
    line_inf, = ax_line.plot([], [], color='red', label='% Infected')
    line_vac, = ax_line.plot([], [], color='green', label='% Vaccinated')
    ax_line.set_xlim(0, total_time)
    ax_line.set_ylim(0, 100)
    ax_line.set_xlabel('Time')
    ax_line.set_ylabel('Percentage')
    ax_line.legend()
    ax_line.grid(True)

    # Time series data
    times, frac_inf, frac_vac = [], [], []

    def update(frame):
        """Update function for animation."""
        nonlocal pos, phi, seg_steps, infected, vaccinated, immune
        nonlocal t_inf, t_immune_max, t_immune_age
        
        # Run multiple steps per frame for faster simulation
        for _ in range(plot_interval):
            t = frame * dt * plot_interval

            # Linear-segment movement
            pos[:,0] += move_scale * np.cos(phi)
            pos[:,1] += move_scale * np.sin(phi)

            # Wall reflections
            for dim in (0, 1):
                # Lower wall
                mask = pos[:,dim] < 0
                if mask.any():
                    pos[mask,dim] = -pos[mask,dim]
                    phi[mask] = (-1)**dim * (np.pi - phi[mask])
                
                # Upper wall
                mask = pos[:,dim] > domain_size
                if mask.any():
                    pos[mask,dim] = 2*domain_size - pos[mask,dim]
                    phi[mask] = (-1)**dim * (np.pi - phi[mask])

            # Wrap angles to [-π,π]
            phi = (phi + np.pi) % (2*np.pi) - np.pi

            # Update motion segments
            seg_steps -= 1
            done = seg_steps <= 0
            if done.any():
                phi[done] = 2*np.pi * np.random.rand(done.sum())
                seg_steps[done] = np.random.randint(
                    min_seg_steps,
                    max_seg_steps+1,
                    size=done.sum()
                )

            # Deposit & diffuse infection field
            P = np.zeros((grid_size, grid_size))
            idx = (pos[infected] * (grid_size/domain_size)).astype(int) % grid_size
            for i,j in idx:
                P[j,i] += 1
            P = np.real(np.fft.ifft2(np.fft.fft2(P) * diff_factor))
            P = np.clip(P, 0, 1)

            # Infection process
            coords = pos.T * (grid_size/domain_size)
            p_sampled = map_coordinates(P, [coords[1], coords[0]],
                                      order=1, mode='wrap')
            p_inf = np.clip(beta_i * p_sampled, 0, 1)
            sus = (~infected) & (~vaccinated) & (~immune)
            new_inf = sus & (np.random.rand(N_agents) < p_inf)
            infected[new_inf] = True
            t_inf[new_inf] = 0.0

            # Recovery & temporary immunity
            t_inf[infected] += dt
            recov = infected & (t_inf >= tau_i)
            if recov.any():
                t_immune_max[recov] = immunity_multiplier * tau_i[recov]
                immune[recov] = True
                t_immune_age[recov] = 0.0
                infected[recov] = False
                t_inf[recov] = 0.0

            # Update immunity timers
            t_immune_age[immune] += dt
            out = immune & (t_immune_age >= t_immune_max)
            if out.any():
                immune[out] = False
                t_immune_age[out] = 0.0
                t_immune_max[out] = 0.0

            # Vaccination station
            d2 = np.sum((pos - station_center)**2, axis=1)
            near = (d2 < station_radius**2) & (~infected) & (~vaccinated)
            vac_ev = near & (np.random.rand(N_agents) < vacc_prob)
            vaccinated[vac_ev] = True

        # Update plot data
        times.append(frame * dt * plot_interval)
        frac_inf.append(infected.mean()*100)
        frac_vac.append(vaccinated.mean()*100)

        # Update colors
        cols = np.zeros((N_agents, 3))
        sus = (~infected & ~immune & ~vaccinated)
        cols[sus] = [0, 0, 1]  # Blue for susceptible
        cols[infected] = [1, 0, 0]  # Red for infected
        cols[immune & ~vaccinated] = [1, 0.5, 0]  # Orange for immune
        cols[vaccinated] = [0, 1, 0]  # Green for vaccinated

        # Update plots
        scat.set_offsets(pos)
        scat.set_facecolors(cols)
        ax_scat.set_title(f"t = {frame * dt * plot_interval:.2f}")

        line_inf.set_data(times, frac_inf)
        line_vac.set_data(times, frac_vac)
        ax_line.set_xlim(0, frame * dt * plot_interval + dt)

        return scat, line_inf, line_vac

    # Create animation
    frames = steps // plot_interval
    anim = FuncAnimation(
        fig, update, frames=frames,
        interval=50, blit=True
    )

    # Save animation as GIF
    anim.save(output_file, writer='pillow', fps=20)
    plt.close()


if __name__ == '__main__':
    run_simulation_and_save_gif(
        total_time=1000.0,  # Shorter simulation time for example
        output_file='disease_spread.gif'
    )