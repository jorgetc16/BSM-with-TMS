#%%
import os, sys
sys.path.append("/home/jortecal/GitHub/TMS/DarkPhoton/")
sys.path.append("/home/jortecal/GitHub/TMS/DarkPhoton/notebooks/")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from grf.grf import FIRAS
from grf.units import *
from grf.pk_interp import PowerSpectrumGridInterpolator

from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# %matplotlib inline
#%%
##Plot parameters
from plot_params import params

for k, v in params.items():
    if k in pylab.rcParams:
        try:
            pylab.rcParams[k] = v
        except (ValueError, TypeError):
            pass

cols_default = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%%
# Initialize FIRAS/probability machinery
os.chdir("/home/jortecal/GitHub/TMS/DarkPhoton/notebooks/")
pspec = PowerSpectrumGridInterpolator("nonlin_matter_bj")
firas = FIRAS(pspec)

#%%
#  Load TMS sensitivity data
dat_path = "/home/jortecal/GitHub/TMS/TMSSensitivity.dat"
tms_data = np.loadtxt(dat_path)
tms_freq_ghz = tms_data[:, 0]       # GHz
tms_delta_f  = tms_data[:, 1]          # W m^-2 Hz^-1 sr^-1

# Physical constants (SI)
h_SI  = 6.62607015e-34
kB_SI = 1.380649e-23
c_SI  = 299792458.0
T_CMB_SI = 2.7255

# CMB blackbody at TMS frequencies
tms_nu_hz = tms_freq_ghz * 1e9
tms_f_gamma = (2.0 * h_SI * tms_nu_hz**3 / c_SI**2) / np.expm1(h_SI * tms_nu_hz / (kB_SI * T_CMB_SI))

# TMS fractional sensitivity: delta_f / f_gamma
tms_sensitivity_ratio = tms_delta_f / tms_f_gamma
print(f"TMS frequency range: {tms_freq_ghz.min():.1f} – {tms_freq_ghz.max():.1f} GHz")
print(f"TMS sensitivity ratio range: {tms_sensitivity_ratio.min():.2e} – {tms_sensitivity_ratio.max():.2e}")

#%%
# Convert TMS frequencies to natural units (angular frequency omega)
# nu [GHz] -> nu [Hz] -> omega = 2*pi*nu [rad/s] -> natural units
tms_omega = 2 * np.pi * tms_nu_hz * Hz  # Hz is 1/Sec in natural units

print(f"TMS omega range: {tms_omega.min():.4e} – {tms_omega.max():.4e} (natural units)")
print(f"TMS omega range: {tms_omega.min()/eV:.4e} – {tms_omega.max()/eV:.4e} eV")

#%%
# Define mass array and compute sensitivity limits

# Mass range: same as the standard FIRAS scan
m_A_ary = np.logspace(-16, -9, 200) * eV  # fewer points for speed

# Fiducial epsilon used internally by the probability code
eps_base = firas.eps_base  # 1e-7

def compute_eps_lim(m_A, tms_omega, eps_base, tms_sensitivity_ratio):
    """Compute epsilon limit for a single mass point."""
    try:
        # Need to re-initialize inside each worker process
        # because FIRAS objects may not be picklable
        pspec_local = PowerSpectrumGridInterpolator("nonlin_matter_bj")
        firas_local = FIRAS(pspec_local)

        result = firas_local.P_tot_perturb(tms_omega, eps_base, m_A)
        P_tot = result[2]
        P_tot = np.asarray(P_tot, dtype=float)

        if np.all(P_tot <= 0) or np.all(np.isnan(P_tot)):
            return np.nan

        P_floor = eps_base**2 * tms_sensitivity_ratio
        valid = P_tot > P_floor
        if not np.any(valid):
            return np.nan

        eps_candidates = eps_base * np.sqrt(tms_sensitivity_ratio[valid] / P_tot[valid])
        return np.nanmin(eps_candidates)

    except Exception:
        return np.nan

# --- Parallel computation ---
n_cores = max(1, cpu_count() - 1)  # leave one core free
print(f"Using {n_cores} cores for parallel computation")

# Change to correct directory before spawning workers
os.chdir("/home/jortecal/GitHub/TMS/DarkPhoton/notebooks/")

worker = partial(compute_eps_lim,
                 tms_omega=tms_omega,
                 eps_base=eps_base,
                 tms_sensitivity_ratio=tms_sensitivity_ratio)

with Pool(n_cores) as pool:
    eps_lim_ary = np.array(
        list(tqdm(pool.imap(worker, m_A_ary), total=len(m_A_ary), desc="Scanning masses"))
    )

print(f"Computed limits for {np.sum(~np.isnan(eps_lim_ary))} / {len(m_A_ary)} mass points")
eps_lim_ary[eps_lim_ary > 1] = np.nan

#%%
# Plot: TMS sensitivity prospects on epsilon vs m_A'

fig, ax = plt.subplots(figsize=(8, 6))

m_A_eV = m_A_ary / eV  # Convert to eV for plotting

# --- Load existing FIRAS constraint (log-normal PDF, 1+delta in [1e-2, 1e2]) ---
firas_constraint = np.genfromtxt(
    "/home/jortecal/GitHub/TMS/DarkPhoton/data/constraints/fiducial_DP_FIRAS_one_plus_delta_1e2.csv",
    delimiter=",", skip_header=1
)
firas_m_eV = firas_constraint[:, 0]  # already in eV
firas_eps  = firas_constraint[:, 1]
firas_mask = ~np.isnan(firas_eps) & (firas_eps < 1)

# --- Load homogeneous FIRAS constraint ---
firas_homo = np.genfromtxt(
    "/home/jortecal/GitHub/TMS/DarkPhoton/data/constraints/fiducial_DP_FIRAS_homo.csv",
    delimiter=",", skip_header=1
)
homo_m_eV = firas_homo[:, 0]  # already in eV
homo_eps   = firas_homo[:, 1]
homo_mask  = ~np.isnan(homo_eps) & (homo_eps < 1)

# --- PIXIE projection: ~1000x better sensitivity than FIRAS => eps scales as sqrt(1/1000) ---
pixie_scale = 1.0 / np.sqrt(1000.0)
pixie_eps = firas_eps * pixie_scale
pixie_mask = ~np.isnan(pixie_eps) & (pixie_eps < 1)

# # --- Existing constraints: Jupiter ---
# jupiter_data = np.loadtxt(
#     "/home/jortecal/GitHub/TMS/DarkPhoton/data/existing_constraints/Jupiter.txt"
# )
# jup_m_eV = jupiter_data[:, 0]
# jup_eps  = jupiter_data[:, 1]

# --- Existing constraints: Dark SRF projection ---
try:
    srf_data = np.loadtxt(
        "/home/jortecal/GitHub/TMS/DarkPhoton/data/existing_constraints/SRF.cvs",
        delimiter=","
    )
    srf_m_eV = srf_data[:, 0]
    srf_eps  = srf_data[:, 1]
    has_srf = True
except Exception:
    has_srf = False

# --- Plot FIRAS constraints ---
ax.plot(firas_m_eV[firas_mask], firas_eps[firas_mask],
        c='tab:red', lw=2.5, label=r'COBE/FIRAS (Log-normal PDF)')
ax.fill_between(firas_m_eV[firas_mask], firas_eps[firas_mask], 1,
                alpha=0.10, color='tab:red')

ax.plot(homo_m_eV[homo_mask], homo_eps[homo_mask],
        c='gray', lw=2.5, ls=':', label=r'COBE/FIRAS (Homogeneous)')
ax.fill_between(homo_m_eV[homo_mask], homo_eps[homo_mask], 1,
                alpha=0.08, color='gray')

# --- Plot PIXIE projection ---
ax.plot(firas_m_eV[pixie_mask], pixie_eps[pixie_mask],
        c='tab:red', lw=1.5, ls='-.', label=r'PIXIE (projection)')

# --- Plot Jupiter ---
# ax.fill(jup_m_eV, jup_eps, alpha=0.2, color='gray', label=r'Jupiter')

# --- Plot Dark SRF ---
if has_srf:
    ax.plot(srf_m_eV, srf_eps, c='tab:orange', lw=1.5, ls='--', label=r'Dark SRF (projection)')

# --- Plot TMS sensitivity ---
mask = ~np.isnan(eps_lim_ary)
ax.plot(m_A_eV[mask], eps_lim_ary[mask], c='tab:blue', lw=2.5, label=r'TMS (projection)')
ax.fill_between(m_A_eV[mask], eps_lim_ary[mask], 1, alpha=0.15, color='tab:blue')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$m_{A^\prime}\,[\mathrm{eV}]$")
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"$\gamma \to A^\prime$")

ax.set_xlim(1e-16, 1e-9)
ax.set_ylim(1e-12, 1E-1)

ax.legend(fontsize=11, loc='lower left')

plt.tight_layout()
plt.savefig("/home/jortecal/GitHub/TMS/DarkPhoton/Figures/DarkPhoton_try.pdf", bbox_inches='tight')
plt.show()

# %%
