# plot_infection.py
# takes 15sec to appear for total_time=1000
import matplotlib.pyplot as plt
import numpy as np
from simmodule import simulate

def live_demo(**sim_kwargs):
    # run simulation and grab full history
    (times, frac_inf, frac_vac,
     pos_hist, inf_hist, vac_hist, imm_hist) = simulate(
        **sim_kwargs, record_history=True
    )

    # ── figure set-up ───────────────────────────────────────────────
    plt.ion()
    fig, (ax_scat, ax_line) = plt.subplots(2, 1, figsize=(6, 9))

    scat = ax_scat.scatter([], [], s=30)
    ax_scat.set_xlim(0, sim_kwargs['domain_size'])
    ax_scat.set_ylim(0, sim_kwargs['domain_size'])
    ax_scat.set_aspect('equal', 'box')
    ax_scat.set_xticks([]); ax_scat.set_yticks([])

    # vaccination-station marker
    station = plt.Circle(
        np.asarray(sim_kwargs.get('station_rel', (0.9, 0.1))) *
        sim_kwargs['domain_size'],
        sim_kwargs.get('station_radius', 0.1),
        edgecolor='grey', facecolor='none', lw=2, alpha=0.7
    )
    ax_scat.add_patch(station)

    line_inf, = ax_line.plot([], [], 'r-', label='% Infected')
    line_vac, = ax_line.plot([], [], 'g-', label='% Vaccinated')
    ax_line.set_xlim(0, times[-1])
    ax_line.set_ylim(0, 100)
    ax_line.set_xlabel('Time')
    ax_line.set_ylabel('Percentage')
    ax_line.legend()
    ax_line.grid(True)

    # ── animation loop ─────────────────────────────────────────────
    for k, t in enumerate(times):
        pos = pos_hist[k]
        infected = inf_hist[k]
        vaccinated = vac_hist[k]
        immune = imm_hist[k]

        colors = np.zeros((len(pos), 3))
        susceptible = (~infected) & (~immune) & (~vaccinated)
        colors[susceptible] = [0, 0, 1]  # blue
        colors[infected] = [1, 0, 0]  # red
        colors[immune & ~vaccinated] = [1, 0.5, 0]  # orange
        colors[vaccinated] = [0, 1, 0]  # green

        scat.set_offsets(pos)
        scat.set_facecolors(colors)
        ax_scat.set_title(f"t = {t:.2f}")

        line_inf.set_data(times[:k+1], frac_inf[:k+1])
        line_vac.set_data(times[:k+1], frac_vac[:k+1])
        ax_line.set_xlim(0, t)

        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    live_demo(
        original_N=200,
        domain_size=2.0,
        grid_size=64,
        D=0.1,
        dt=0.01,
        total_time=1000.0,
        move_scale=0.0001,
        plot_interval=50,
        recovery_time=30.0,
        recovery_spread=0.2,
        immunity_multiplier=1.0,
        station_rel=(0.9, 0.1),
        station_radius=0.1,
        vacc_prob=0.05,
        beta=0.005,
        beta_spread=0.2,
        min_seg_steps=50,
        max_seg_steps=600
    )