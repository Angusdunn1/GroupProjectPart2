# simmodule.py
#!/usr/bin/env python3
import numpy as np
from scipy.ndimage import map_coordinates

"""
Agent based Fast Fourier Transfer simulation of virus transmission.
This module defines the nature of spread with parameters given below.

-Simulation visualisation is performed in PlotInfection.py
-Results/plots are made in Sweep.py

The provided code uses Fast Fourier Transfer to simulate viral spread through a 
diffusion model. The simulation is ran in two dimensions. Agents (dots) move in 
straight-line segments of a randomly assigned number of steps in random directions.

Features include:
    Transmissibility and recovery times
    Diffusion of the infection field
    Bernoulli draws for infection trials
    Temporary immunity after recovery. Permanent recover if vaccinated.
"""

def simulate(
    original_N=200, # 200 agents in the area given
    domain_size=2.0, # 2.0 size of square
    grid_size=64, # 64 resolution of the grid
    D=0.1, # 0.1 diffusion coefficient 
    dt=0.01, # 0.01 time step
    total_time=100000.0, # 100000.0 time to run
    move_scale=0.0001, # 0.0001 step length per dt
    plot_interval=50, # 50 
    recovery_time=30.0, # 30 mean infection duration
    recovery_spread=0.2,  #0.2 ±20% variability in duration
    immunity_multiplier=1.0, # 1.0 immune = multiplier × individual recovery_time so you cannot contract the virus immediately again
    station_rel=(0.9, 0.1), # (0.9, 0.1) vaccination station located at bottom right
    station_radius=0.1, # 0.1 station radius
    vacc_prob=0.05, #0.05 per step prob of becoming vaccinated
    beta=0.005, #0.005 mean transmissibility (rate of becoming infected near another infected)
    beta_spread=0.2, #0.2 ±20% variability in beta
    min_seg_steps=50, #50 min. length of straight segment (steps taken)
    max_seg_steps=600, #600 max. length of straight segment
    record_history=False,
    rng=None, 
):
    # initialise a random number generator
    rng = np.random.default_rng(rng)
    
    # compute total steps and number of agents
    steps     = int(total_time / dt)
    N_agents  = int(original_N * domain_size ** 2)

    # uniform random transmissibility for each agent
    beta_i = rng.uniform((1 - beta_spread) * beta,
                         (1 + beta_spread) * beta,
                         size=N_agents)
    # uniform random recovery times for each agent
    tau_i  = rng.uniform((1 - recovery_spread) * recovery_time,
                         (1 + recovery_spread) * recovery_time,
                         size=N_agents)

    # MOTION
    # random positions [0, domain_size]^2
    pos       = rng.random((N_agents, 2)) * domain_size
    # random directions [0, 2π]
    phi       = rng.random(N_agents) * 2 * np.pi
    # random n of steps before turning
    seg_steps = rng.integers(min_seg_steps, max_seg_steps + 1,
                             size=N_agents)

    # INFECTION STATUS
    infected       = np.zeros(N_agents, bool) # flag as infected
    infected[0] = True                      # seed one infected agent
    vaccinated     = np.zeros(N_agents, bool) # flag as vaccinated
    immune         = np.zeros(N_agents, bool) # flag temporary immunity
    t_inf          = np.zeros(N_agents) # time since infection
    t_immune_max   = np.zeros(N_agents) 
    t_immune_age   = np.zeros(N_agents) # time spent immune

    # VACCINATION STATION
    station_center = np.asarray(station_rel) * domain_size

    # SPECTRAL DIFFUSION
    # computing diffusion via numpy fft
    dx  = domain_size / grid_size
    k   = np.fft.fftfreq(grid_size, d=dx) * 2 * np.pi
    k2  = -(k[:, None] ** 2 + k[None, :] ** 2) # negative Laplacian
    diff_factor = np.exp(D * k2 * dt)

    # INIT
    times      = []
    frac_inf   = []
    frac_vac   = []

    pos_hist = []; inf_hist = []; vac_hist = []; imm_hist = []

    # SIMULATION LOOP
    for step in range(1, steps + 1):
        t = step * dt

        # 1) linear-segment motion, reflecting off of walls
        pos[:, 0] += move_scale * np.cos(phi)
        pos[:, 1] += move_scale * np.sin(phi)
        # reflect at boundaries 
        for dim in (0, 1):
            neg = pos[:, dim] < 0
            pos[neg, dim] = -pos[neg, dim]
            phi[neg] = (-1) ** dim * (np.pi - phi[neg])

            pos_dim_max = domain_size
            pos_gt = pos[:, dim] > pos_dim_max
            pos[pos_gt, dim] = 2 * pos_dim_max - pos[pos_gt, dim]
            phi[pos_gt] = (-1) ** dim * (np.pi - phi[pos_gt])

        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        seg_steps -= 1
        finished   = seg_steps <= 0
        if finished.any():
            phi[finished]       = rng.random(finished.sum()) * 2 * np.pi
            seg_steps[finished] = rng.integers(min_seg_steps, max_seg_steps + 1,
                                               size=finished.sum())

        # 2) deposit infection strength on grid
        P   = np.zeros((grid_size, grid_size))
        # infected positions to grid indices
        idx = (pos[infected] * (grid_size / domain_size)).astype(int) % grid_size
        for i, j in idx:
            P[j, i] += 1
        # diffuse the field via FFT [0,1]
        P = np.real(np.fft.ifft2(np.fft.fft2(P) * diff_factor))
        P = np.clip(P, 0, 1)

        # 3) infection of susceptible agents
        coords   = pos.T * (grid_size / domain_size)
        p_sample = map_coordinates(P, [coords[1], coords[0]],
                                   order=1, mode='wrap')
        # per agent infection probability 
        p_inf    = np.clip(beta_i * p_sample, 0, 1)

        susceptible   = (~infected) & (~vaccinated) & (~immune)
        # determing new infection with Bernoulli trial
        new_infected  = susceptible & (rng.random(N_agents) < p_inf)
        infected[new_infected] = True
        t_inf[new_infected]    = 0.0 # reset infection timer

        # 4) recovery leads to temporary immunity
        t_inf[infected] += dt
        recovered = infected & (t_inf >= tau_i)
        if recovered.any():
            infected[recovered]      = False
            immune[recovered]        = True
            # set immunity duration per agent
            t_immune_max[recovered]  = immunity_multiplier * tau_i[recovered]
            t_immune_age[recovered]  = 0.0
            t_inf[recovered]         = 0.0
        # remove immunity after a time
        t_immune_age[immune] += dt
        immunity_ended = immune & (t_immune_age >= t_immune_max)
        if immunity_ended.any():
            immune[immunity_ended]       = False
            t_immune_age[immunity_ended] = 0.0
            t_immune_max[immunity_ended] = 0.0

        # 5) vaccination at the bottom right station
        d2        = ((pos - station_center) ** 2).sum(axis=1)
        near      = (d2 < station_radius ** 2) & (~infected) & (~vaccinated)
        new_vaccs = near & (rng.random(N_agents) < vacc_prob)
        vaccinated[new_vaccs] = True

        # 6) bookkeeping every plot_interval
        if step % plot_interval == 0:
            times.append(t)
            frac_inf.append(infected.mean() * 100)
            frac_vac.append(vaccinated.mean() * 100)

            if record_history:
                pos_hist.append(pos.copy())
                inf_hist.append(infected.copy())
                vac_hist.append(vaccinated.copy())
                imm_hist.append(immune.copy())

    if record_history:
        return (np.asarray(times), np.asarray(frac_inf), np.asarray(frac_vac),
                pos_hist, inf_hist, vac_hist, imm_hist)
    else:
        return (np.asarray(times), np.asarray(frac_inf), np.asarray(frac_vac))
    