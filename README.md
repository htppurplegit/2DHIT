# 2D Pseudo-Spectral HIT Solver

Solves the 2D incompressible Navier–Stokes equation in vorticity form:

```
∂ω/∂t + u·∇ω = ν∇²ω + f
```

Velocity is recovered from the streamfunction ψ: `u = ∂ψ/∂y`, `v = −∂ψ/∂x`, `ω = −∇²ψ`

| Feature | Details |
|---|---|
| Spatial discretisation | Pseudo-spectral (FFT), 2/3-rule dealiasing |
| Time integration | 4th-order Runge–Kutta, CFL-adaptive dt |
| Forcing | Negative-damping: fixed energy injection rate `eps_f` in band `[k_lo, k_hi]` |
| Boundary conditions | Doubly-periodic `[0, L)²` |

---

## Dependencies

```bash
pip install numpy matplotlib
```

Python ≥ 3.11 (uses `tomllib` from the standard library).

---

## Running

```bash
# Fresh run (reads params.toml automatically)
python hit2d.py

# Custom config file
python hit2d.py --config my_run.toml

# Restart from a checkpoint
python hit2d.py --restart results/checkpoints/cp_final.npz
```

---

## Parameters — `params.toml`

```toml
[simulation]
N      = 256        # Grid points per side (power of 2: 128 / 256 / 512)
L      = 6.2831853  # Domain length = 2π
nu     = 1.0e-4     # Kinematic viscosity
T      = 30.0       # Simulation end time
t_out  = 0.5        # Output interval
seed   = 42         # Random seed
CFL    = 0.5        # CFL safety factor
dt_max = 1.0e-3     # Maximum dt
dt_min = 1.0e-8     # Minimum dt

[initial_conditions]
k0 = 4.0    # Peak wavenumber of initial spectrum
E0 = 0.5    # Target initial kinetic energy

[forcing]
k_lo  = 3.0    # Lower wavenumber of forcing band
k_hi  = 5.0    # Upper wavenumber of forcing band
eps_f = 0.1    # Energy injection rate [energy/time]

[checkpoint]
enabled  = true
interval = 2.0   # Checkpoint every N time units
restart  = ""    # Path to restart file (empty = fresh start)

[output]
results_dir = "results"
```

---

## Forcing

The **negative-damping** method injects energy at exactly `eps_f` [energy/time] into modes in the band `[k_lo, k_hi]`:

```
f̂(k) = eps_f / (2 E_band) · mask(k) · ω̂(k)
```

where `E_band` is the kinetic energy in the forcing band. Evaluated at every RK4 stage.

> **Note:** the forcing band must already contain energy at the start of the run.
> Use initial conditions with `k0` inside or near `[k_lo, k_hi]`.

---

## Output

All files are written to `results/` (created automatically):

```
results/
├── params.toml             ← copy of config used
├── hit2d_results.png       ← 4-panel summary figure
├── diagnostics.npz         ← time series: t, E(t), Z(t)
├── spectrum_initial.npz    ← E(k) at t = 0
├── spectrum_final.npz      ← E(k) at t = T
└── checkpoints/
    ├── cp_t002.000.npz
    └── cp_final.npz
```

### Summary figure panels

| Panel | Contents |
|---|---|
| Top-left | Vorticity field ω(x, y) at final time |
| Top-right | Energy spectrum E(k) with k⁻³ reference, initial spectrum, and forcing band |
| Bottom-left | Kinetic energy E(t) |
| Bottom-right | Enstrophy Z(t) |

---

## Loading results

```python
import numpy as np
import matplotlib.pyplot as plt

# Time series
diag = np.load("results/diagnostics.npz")
plt.plot(diag["t"], diag["E"], label="KE")
plt.plot(diag["t"], diag["Z"], label="Enstrophy")

# Energy spectra
sp0 = np.load("results/spectrum_initial.npz")
spf = np.load("results/spectrum_final.npz")
plt.loglog(sp0["k"][1:], sp0["Ek"][1:], "--", label="t = 0")
plt.loglog(spf["k"][1:], spf["Ek"][1:],  "-", label="t = T")

# Restart field
ck = np.load("results/checkpoints/cp_final.npz")
omega_hat = ck["omega_hat_re"] + 1j * ck["omega_hat_im"]
```

---

## Physics background

2D turbulence has two inviscid conserved quantities:

- **Kinetic energy** `E` — cascades to **large** scales (inverse cascade)
- **Enstrophy** `Z` — cascades to **small** scales (direct cascade)

Key spectral signatures:

| Range | Slope | Physics |
|---|---|---|
| k < k_forcing | E(k) ~ k⁻⁵/³ | Inverse energy cascade |
| k > k_forcing | E(k) ~ k⁻³ | Direct enstrophy cascade |
