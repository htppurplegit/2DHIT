#!/usr/bin/env python3
"""
2D Pseudo-Spectral HIT Solver
==============================
Solves the 2D incompressible Navier-Stokes equation in vorticity form:

    dw/dt + u·grad w = ν grad^2 w + f

Velocity is recovered from the streamfunction phi:
    u = d phi/dy,  v = −d phi/dx,  w = −grad^2 phi

Forcing: negative-damping — injects energy at a fixed rate eps_f
         into spectral modes in the band [k_lo, k_hi].

    ^f(k) = (eps_f / 2 E_band) · mask(k) · ^omega(k)

Spatial discretisation : pseudo-spectral (FFT), 2/3-rule dealiasing
Time integration       : 4th-order Runge-Kutta, CFL-adaptive dt
Boundary conditions    : doubly-periodic [0, L)²

Usage:
    python hit2d.py                              # reads params.toml
    python hit2d.py --config my_run.toml         # custom config
    python hit2d.py --restart results/checkpoints/cp_final.npz

Dependencies:
    pip install numpy matplotlib
    Python >= 3.11  (tomllib is in the standard library)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tomllib
except ImportError:
    sys.exit("tomllib not found. Use Python >= 3.11 or: pip install tomli")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

cli = argparse.ArgumentParser(description="2D pseudo-spectral HIT solver")
cli.add_argument("--config",  default="params.toml", help="TOML parameter file")
cli.add_argument("--restart", default=None,           help="Path to .npz checkpoint")
args = cli.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

cfg_path = Path(args.config)
if not cfg_path.exists():
    sys.exit(f"Config not found: {cfg_path}")

with open(cfg_path, "rb") as fh:
    cfg = tomllib.load(fh)

sim = cfg["simulation"]
ic  = cfg["initial_conditions"]
frc = cfg["forcing"]
ck  = cfg["checkpoint"]
out = cfg["output"]

N       = int(sim["N"])
L       = float(sim["L"])
nu      = float(sim["nu"])
T       = float(sim["T"])
t_out   = float(sim["t_out"])
SEED    = int(sim["seed"])
CFL     = float(sim["CFL"])
dt_max  = float(sim["dt_max"])
dt_min  = float(sim["dt_min"])

k0    = float(ic["k0"])
E0    = float(ic["E0"])

k_lo  = float(frc["k_lo"])
k_hi  = float(frc["k_hi"])
eps_f = float(frc["eps_f"])

ckpt_enabled  = bool(ck["enabled"])
ckpt_interval = float(ck["interval"])
restart_file  = args.restart or (ck.get("restart", "").strip() or None)

results_dir = Path(out["results_dir"])

# ══════════════════════════════════════════════════════════════════════════════
# GRID AND WAVENUMBER ARRAYS
# ══════════════════════════════════════════════════════════════════════════════

dx = L / N
x  = np.arange(N) * dx

kx = fftfreq(N, d=1.0 / N)
ky = fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(kx, ky, indexing="ij")

K2  = KX**2 + KY**2
K   = np.sqrt(K2)
K2s = K2.copy()
K2s[0, 0] = 1.0          # avoid division by zero at DC mode

alpha  = 2.0 * np.pi / L
alpha2 = alpha**2

kmax    = N // 3          # 2/3-rule dealiasing cutoff
dealias = ((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)).astype(float)
f_mask  = ((K >= k_lo)  & (K  <= k_hi)).astype(float)

rng = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def kinetic_energy(omega_hat: np.ndarray) -> float:
    """E = (1/(2 α² N⁴)) Σ |ω̂|²/|k|²"""
    return 0.5 * float(np.sum(np.abs(omega_hat)**2 / K2s)) / (alpha2 * N**4)


def enstrophy(omega_hat: np.ndarray) -> float:
    """Z = (1/(2 N⁴)) Σ |ω̂|²"""
    return 0.5 * float(np.sum(np.abs(omega_hat)**2)) / N**4


def energy_spectrum(omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shell-averaged 1D energy spectrum E(k)."""
    ke_mode = 0.5 * np.abs(omega_hat)**2 / K2s / (alpha2 * N**4)
    K_int   = np.round(K).astype(int).ravel()
    k_max   = N // 2
    valid   = K_int <= k_max
    E_k     = np.bincount(K_int[valid], weights=ke_mode.ravel()[valid],
                          minlength=k_max + 1)
    return np.arange(k_max + 1, dtype=float), E_k

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE TIME STEP
# ══════════════════════════════════════════════════════════════════════════════

def compute_dt(omega_hat: np.ndarray) -> float:
    psi_hat       = -omega_hat / (alpha2 * K2s)
    psi_hat[0, 0] = 0.0
    u     = np.real(ifft2( 1j * alpha * KY * psi_hat))
    v     = np.real(ifft2(-1j * alpha * KX * psi_hat))
    u_max = max(float(np.abs(u).max()), float(np.abs(v).max()), 1e-10)
    dt    = CFL * dx / u_max
    if dt < dt_min:
        print(f"  WARNING: CFL dt = {dt:.2e} < dt_min = {dt_min:.2e}. Clamping.")
    return float(np.clip(dt, dt_min, dt_max))

# ══════════════════════════════════════════════════════════════════════════════
# FORCING  (negative-damping)
# ══════════════════════════════════════════════════════════════════════════════

def compute_forcing(omega_hat: np.ndarray) -> np.ndarray:
    """
    Inject energy at rate eps_f [energy/time] into wavenumber band [k_lo, k_hi].

    f̂(k) = eps_f / (2 E_band) · mask(k) · ω̂(k)

    where E_band = kinetic energy contained in the forcing band.
    Evaluated inside every RK4 stage for accuracy.
    """
    E_band = 0.5 * float(np.sum(f_mask * np.abs(omega_hat)**2 / K2s)) / (alpha2 * N**4)
    if E_band < 1e-30:
        return np.zeros_like(omega_hat)
    return (eps_f / (2.0 * E_band)) * f_mask * omega_hat

# ══════════════════════════════════════════════════════════════════════════════
# RHS
# ══════════════════════════════════════════════════════════════════════════════

def rhs(omega_hat: np.ndarray) -> np.ndarray:
    """dω̂/dt = −F[u ∂ω/∂x + v ∂ω/∂y] − ν α² |k|² ω̂ + f̂"""
    psi_hat       = -omega_hat / (alpha2 * K2s)
    psi_hat[0, 0] = 0.0

    u_hat    =  1j * alpha * KY * psi_hat
    v_hat    = -1j * alpha * KX * psi_hat
    domx_hat =  1j * alpha * KX * omega_hat
    domy_hat =  1j * alpha * KY * omega_hat

    u    = np.real(ifft2(dealias * u_hat))
    v    = np.real(ifft2(dealias * v_hat))
    domx = np.real(ifft2(dealias * domx_hat))
    domy = np.real(ifft2(dealias * domy_hat))

    nl_hat   = fft2(u * domx + v * domy)
    visc_hat = nu * alpha2 * K2 * omega_hat

    return -nl_hat - visc_hat + compute_forcing(omega_hat)

# ══════════════════════════════════════════════════════════════════════════════
# RK4
# ══════════════════════════════════════════════════════════════════════════════

def rk4_step(omega_hat: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(omega_hat)
    k2 = rhs(omega_hat + 0.5 * dt * k1)
    k3 = rhs(omega_hat + 0.5 * dt * k2)
    k4 = rhs(omega_hat +       dt * k3)
    return omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ══════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def init_vorticity() -> np.ndarray:
    """Random vorticity field with E(k) ~ k³ exp(−2(k/k0)²), scaled to E0."""
    amp       = K**2 * np.exp(-(K / k0)**2)
    omega_hat = amp * np.exp(1j * 2.0 * np.pi * rng.random((N, N)))
    omega_hat = fft2(np.real(ifft2(omega_hat)))   # enforce real physical field
    omega_hat *= np.sqrt(E0 / kinetic_energy(omega_hat))
    return omega_hat

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(omega_hat: np.ndarray, t: float, step: int,
                    history: dict, tag: str | None = None) -> Path:
    path = ckpt_dir / f"cp_{tag or f't{t:09.3f}'}.npz"
    np.savez_compressed(
        path,
        omega_hat_re = omega_hat.real,
        omega_hat_im = omega_hat.imag,
        t            = [t],
        step         = [step],
        hist_t       = history["t"],
        hist_E       = history["E"],
        hist_Z       = history["Z"],
    )
    return path


def load_checkpoint(path: str | Path) -> tuple:
    d = np.load(path)
    omega_hat = d["omega_hat_re"] + 1j * d["omega_hat_im"]
    if omega_hat.shape != (N, N):
        sys.exit(f"ERROR: checkpoint shape {omega_hat.shape} != ({N}, {N}). "
                 f"Ensure N in params.toml matches the checkpoint.")
    t       = float(d["t"][0])
    step    = int(d["step"][0])
    history = {"t": list(d["hist_t"]), "E": list(d["hist_E"]), "Z": list(d["hist_Z"])}
    return omega_hat, t, step, history

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

ckpt_dir = results_dir / "checkpoints"
results_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(cfg_path, results_dir / cfg_path.name)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run() -> tuple:
    print("=" * 60)
    print("  2D Pseudo-Spectral HIT Solver")
    print(f"  Grid    : {N}×{N}   L = {L:.4f}")
    print(f"  Physics : ν = {nu}   CFL = {CFL}")
    print(f"  Forcing : negative_damping   eps_f = {eps_f}"
          f"   band = [{k_lo}, {k_hi}]")
    print(f"  Results : {results_dir.resolve()}")
    print("=" * 60)

    # Initialise or restart
    if restart_file:
        print(f"\nRestarting from: {restart_file}")
        omega_hat, t, step, history = load_checkpoint(restart_file)
        t_next = t + t_out
    else:
        omega_hat = init_vorticity()
        t, step   = 0.0, 0
        history   = {"t": [], "E": [], "Z": []}
        t_next    = 0.0
        k_init, Ek_init = energy_spectrum(omega_hat)
        np.savez(results_dir / "spectrum_initial.npz", k=k_init, Ek=Ek_init)

    t_next_ckpt = (t // ckpt_interval + 1) * ckpt_interval if ckpt_enabled else float("inf")

    hdr = f"{'Step':>9}  {'t':>9}  {'KE':>14}  {'Enstrophy':>14}  {'dt':>10}"
    print(hdr)
    print("─" * len(hdr))

    dt     = compute_dt(omega_hat)
    t_wall = time.time()

    while t < T - 1e-12:

        # Diagnostic output
        if t >= t_next - 1e-12:
            E = kinetic_energy(omega_hat)
            Z = enstrophy(omega_hat)
            history["t"].append(t)
            history["E"].append(E)
            history["Z"].append(Z)
            print(f"{step:>9d}  {t:>9.3f}  {E:>14.6e}  {Z:>14.6e}  {dt:>10.3e}")
            t_next += t_out

        # Checkpoint
        if t >= t_next_ckpt - 1e-12:
            cp = save_checkpoint(omega_hat, t, step, history)
            print(f"  checkpoint → {cp.name}")
            t_next_ckpt += ckpt_interval

        # Advance
        omega_hat  = rk4_step(omega_hat, dt)
        omega_hat *= dealias    # enforce 2/3-rule dealiasing
        t    += dt
        step += 1
        dt    = compute_dt(omega_hat)

    elapsed = time.time() - t_wall
    per_step = f"  ({elapsed/step*1e3:.2f} ms/step)" if step > 0 else ""
    print(f"\nCompleted {step} steps in {elapsed:.1f}s{per_step}")

    if ckpt_enabled:
        cp = save_checkpoint(omega_hat, t, step, history, tag="final")
        print(f"Final checkpoint → {cp}")

    diag_path = results_dir / "diagnostics.npz"
    np.savez(diag_path,
             t=np.array(history["t"]),
             E=np.array(history["E"]),
             Z=np.array(history["Z"]))
    print(f"Diagnostics      → {diag_path}")

    k_fin, Ek_fin = energy_spectrum(omega_hat)
    np.savez(results_dir / "spectrum_final.npz", k=k_fin, Ek=Ek_fin)
    print(f"Final spectrum   → {results_dir / 'spectrum_final.npz'}")

    return omega_hat, history

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(omega_hat: np.ndarray, history: dict) -> None:
    k_arr, E_k = energy_spectrum(omega_hat)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"2D HIT  |  N={N},  ν={nu},  t_end={history['t'][-1]:.1f}",
                 fontsize=13)

    # Vorticity
    ax = axes[0, 0]
    X_g, Y_g = np.meshgrid(x, x, indexing="ij")
    omega = np.real(ifft2(omega_hat))
    vlim  = np.percentile(np.abs(omega), 99)
    im    = ax.pcolormesh(X_g.T, Y_g.T, omega.T, cmap="RdBu_r",
                          vmin=-vlim, vmax=vlim, shading="auto")
    plt.colorbar(im, ax=ax, label="ω")
    ax.set_title(f"Vorticity  (t = {history['t'][-1]:.2f})")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

    # Energy spectrum
    ax = axes[0, 1]
    kp, Ep = k_arr[1:], E_k[1:]
    mask = kp <= kmax
    kp, Ep = kp[mask], Ep[mask]
    ax.loglog(kp, Ep, "b-", lw=2, label="E(k) — final")
    init_sp = results_dir / "spectrum_initial.npz"
    if init_sp.exists():
        d0 = np.load(init_sp)
        k0p, Ek0p = d0["k"][1:], d0["Ek"][1:]
        m0 = k0p <= kmax
        ax.loglog(k0p[m0], Ek0p[m0], "b--", lw=1, alpha=0.4, label="E(k) — initial")
    ref_mask = (kp >= k0) & (kp <= kmax)
    if ref_mask.sum() > 1:
        i_ref = np.argmax(ref_mask)
        C = Ep[i_ref] * kp[i_ref]**3
        ax.loglog(kp[ref_mask], C * kp[ref_mask]**(-3), "k--", lw=1.5, label="k⁻³")
    ax.axvline(k0,   color="gray", ls=":", lw=1, label=f"k₀={k0:.0f}")
    ax.axvline(kmax, color="red",  ls=":", lw=1, label=f"k_dA={kmax}")
    ax.axvspan(k_lo, k_hi, alpha=0.12, color="green",
               label=f"Forcing [{k_lo:.0f}–{k_hi:.0f}]")
    ax.set_xlim(1, kmax); ax.set_ylim(bottom=1e-11)
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.set_title("Energy spectrum")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Kinetic energy
    ax = axes[1, 0]
    ax.plot(history["t"], history["E"], "b-", lw=2)
    ax.set_xlabel("t"); ax.set_ylabel("E"); ax.set_title("KE vs time")
    ax.grid(True, alpha=0.3)

    # Enstrophy
    ax = axes[1, 1]
    ax.plot(history["t"], history["Z"], "r-", lw=2)
    ax.set_xlabel("t"); ax.set_ylabel("Z"); ax.set_title("Enstrophy vs time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = results_dir / "hit2d_results.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved     → {fig_path}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    omega_hat_final, history = run()
    plot_results(omega_hat_final, history)
