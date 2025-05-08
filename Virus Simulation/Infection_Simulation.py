import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import imageio.v2 as imageio  # for GIF export
from io import BytesIO        # to capture frame from plt

"""
This file is used solely to make the gif.
"""

def run_agent_with_temporary_immunity(
    original_N=200,
    domain_size=2.0,
    grid_size=64,
    D=0.1,
    dt=0.01,
    total_time=1000.0,
    steps=50000,
    move_scale=0.001,
    plot_interval=50,
    recovery_time=30.0, # infection duration
    immunity_multiplier=1.0, # immune = multiplier × infection duration
    station_rel=(0.9, 0.1),
    station_radius=0.1,
    vacc_prob=0.05,
    beta=0.005
):
    steps = int(total_time / dt)
    t = 0.0
    N_agents = int(original_N * domain_size**2)

    pos = np.random.rand(N_agents, 2) * domain_size

    infected = np.zeros(N_agents, bool)
    infected[0] = True
    vaccinated = np.zeros(N_agents, bool)
    immune = np.zeros(N_agents, bool)
    t_inf = np.zeros(N_agents, float)
    t_immune_max = np.zeros(N_agents, float)
    t_immune_age = np.zeros(N_agents, float)

    station_center = np.array(station_rel) * domain_size

    dx = domain_size / grid_size
    k = np.fft.fftfreq(grid_size, d=dx) * 2*np.pi
    k2 = -(k[:,None]**2 + k[None,:]**2)
    diff_factor = np.exp(D * k2 * dt)

    plt.ioff()
    fig, (ax_scat, ax_line) = plt.subplots(2, 1, figsize=(6,9))

    scat = ax_scat.scatter(pos[:,0], pos[:,1], s=30)
    station = plt.Circle(station_center, station_radius, edgecolor='grey',
                         facecolor='none', lw=2, alpha=0.7)
    ax_scat.add_patch(station)
    ax_scat.set_xlim(0, domain_size)
    ax_scat.set_ylim(0, domain_size)
    ax_scat.set_aspect('equal', adjustable='box')
    ax_scat.set_xticks([]); ax_scat.set_yticks([])

    line_inf, = ax_line.plot([], [], label='% Infected')
    line_vac, = ax_line.plot([], [], label='% Vaccinated')
    ax_line.set_xlim(0, steps*dt)
    ax_line.set_ylim(0, 100)
    ax_line.set_xlabel('Time')
    ax_line.set_ylabel('Percentage')
    ax_line.legend()
    ax_line.grid(True)

    times, frac_inf, frac_vac = [], [], []

    frames = []  # collect frames for GIF

    for n in range(1, steps+1):
        pos += move_scale * np.random.randn(N_agents, 2)
        pos %= domain_size

        P = np.zeros((grid_size, grid_size))
        idx = (pos[infected] * grid_size / domain_size).astype(int) % grid_size
        for i,j in idx:
            P[j, i] += 1
        P = np.real(np.fft.ifft2(np.fft.fft2(P) * diff_factor))
        P = np.clip(P, 0, 1)

        coords = pos.T * (grid_size / domain_size)
        p_sampled = map_coordinates(P, [coords[1], coords[0]],
                                    order=1, mode='wrap')
        p_inf = np.clip(beta * p_sampled, 0, 1)
        can_be_inf = (~infected) & (~vaccinated) & (~immune)
        new_inf = can_be_inf & (np.random.rand(N_agents) < p_inf)
        infected[new_inf] = True
        t_inf[new_inf] = 0.0

        t_inf[infected] += dt
        recov = infected & (t_inf >= recovery_time)
        if recov.any():
            t_immune_max[recov] = immunity_multiplier * t_inf[recov]
            immune[recov] = True
            t_immune_age[recov] = 0.0
            infected[recov] = False
            t_inf[recov] = 0.0

        t_immune_age[immune] += dt
        out_of_immunity = immune & (t_immune_age >= t_immune_max)
        immune[out_of_immunity] = False
        t_immune_age[out_of_immunity] = 0.0
        t_immune_max[out_of_immunity] = 0.0

        d2 = np.sum((pos - station_center)**2, axis=1)
        near = (d2 < station_radius**2) & (~infected) & (~vaccinated)
        vac_ev = near & (np.random.rand(N_agents) < vacc_prob)
        vaccinated[vac_ev] = True

        if n % plot_interval == 0:
            t = n * dt
            times.append(t)
            frac_inf.append(infected.mean()*100)
            frac_vac.append(vaccinated.mean()*100)

            colours = np.zeros((N_agents, 3))
            mask_s = (~infected & ~immune & ~vaccinated)
            colours[mask_s] = [0, 0, 1]           # susceptible
            colours[infected] = [1, 0, 0]         # infected
            colours[immune & ~vaccinated] = [1, 0.5, 0]  # immune
            colours[vaccinated] = [0, 1, 0]       # vaccinated

            scat.set_offsets(pos)
            scat.set_facecolors(colours)
            ax_scat.set_title(f"t = {t:.2f}")

            line_inf.set_data(times, frac_inf)
            line_vac.set_data(times, frac_vac)
            ax_line.set_xlim(0, t + dt)

            # Capture frame
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

    plt.close(fig)  # close figure to avoid duplicate display

    # Save GIF
    imageio.mimsave('infection_simulation.gif', frames, fps=50)
    print("Saved animation as 'infection_simulation.gif'")

if __name__ == '__main__':
    run_agent_with_temporary_immunity()
