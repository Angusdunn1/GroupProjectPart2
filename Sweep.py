# sweep.py
# takes 5min to appear for total_time=60
import numpy as np
import matplotlib.pyplot as plt
from Simmodule import simulate

# ── parameters ──────────────────────────────────────────────────────
betas = np.linspace(0.001, 0.01, 10)
taus = np.linspace(10, 60, 10)
move_scales = np.linspace(0.00001, 0.001, 10)

# ── allocate arrays ─────────────────────────────────────────────────
Burden = np.zeros((len(betas), len(taus)))
T10 = np.zeros_like(Burden)
PeakTime = np.zeros_like(Burden)

# ── β × τ sweep ──────────────────────────────────────────────────────
for i, b in enumerate(betas):
    for j, tau in enumerate(taus):
        times, inf_ts, _ = simulate(
            beta=b,
            recovery_time=tau,
            total_time=60.0,
            dt=0.01,
            original_N=200
        )
        dt_eff = times[1] - times[0]
        Burden[i,j] = inf_ts.sum() * dt_eff
        T10[i,j] = (inf_ts >= 5.0).sum() * dt_eff
        PeakTime[i,j] = times[np.argmax(inf_ts)]

# ── plot: Burden ─────────────────────────────────────────────────────
plt.figure()
plt.imshow(
    Burden.T,
    origin='lower',
    extent=[betas[0], betas[-1], taus[0], taus[-1]],
    aspect='auto'
)
plt.colorbar(label='Integrated burden (% infected·time)')
plt.xlabel('β')
plt.ylabel('Recovery time τ')
plt.title('Area under % infected curve')
plt.show()

# ── plot: PeakTime ──────────────────────────────────────────────────
plt.figure()
plt.imshow(
    PeakTime.T,
    origin='lower',
    extent=[betas[0], betas[-1], taus[0], taus[-1]],
    aspect='auto'
)
plt.colorbar(label='Time of peak infection')
plt.xlabel('β')
plt.ylabel('Recovery time τ')
plt.title('Peak infection time')
plt.show()

# ── β × move_scale sweep for PeakHeight ─────────────────────────────
PeakHt = np.zeros((len(betas), len(move_scales)))

for i, b in enumerate(betas):
    for j, ms in enumerate(move_scales):
        times, inf_ts, _ = simulate(
            beta=b,
            move_scale=ms,
            total_time=200.0,
            dt=0.05,
            original_N=50
        )
        PeakHt[i,j] = inf_ts.max()

# ── plot: PeakHeight vs β & move_scale ──────────────────────────────
plt.figure()
plt.imshow(
    PeakHt.T,
    origin='lower',
    extent=[betas[0], betas[-1], move_scales[0], move_scales[-1]],
    aspect='auto'
)
plt.colorbar(label='Max % infected')
plt.xlabel('β')
plt.ylabel('move_scale')
plt.title('Peak outbreak size vs mobility & transmissibility')
plt.xscale('linear')
plt.yscale('log')
plt.show()
print("plots take a while to load (maybe 10min)!")
