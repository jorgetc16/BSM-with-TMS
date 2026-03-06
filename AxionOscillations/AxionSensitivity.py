"""
Compute TMS sensitivity to axion-photon coupling for a range of axion masses.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import matplotlib.patheffects as path_effects
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import sys

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "AxionOscillations"))

from TMSSensitivity import sensitivity_interpolator
from plasma import GHz_to_GeV, GeV_to_GHz
from mixing import P_gamma_to_alp

# Plotting setup
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Physical constants
h_SI = 6.62607015e-34
kB_SI = 1.380649e-23
c_SI = 299792458.0
T_CMB_SI = 2.7255

# Unit conversion
eV_to_GeV = 1e-9
GeV_to_eV = 1e9

def galactic_directions(n_directions=100, avoid_poles=True, max_latitude=20):
    """
    Generate uniformly distributed directions across the sky.
    Returns (l, b) in degrees, where l is galactic longitude, b is latitude.
    
    Parameters:
    -----------
    n_directions : int
        Number of directions to generate
    avoid_poles : bool
        If True, restrict to galactic plane region
    max_latitude : float
        Maximum |b| in degrees (only used if avoid_poles=True)
    """
    if avoid_poles:
        # Sample uniformly in galactic plane region where models are most reliable
        # Uniform in sin(b) for uniform coverage
        sin_b_max = np.sin(np.radians(max_latitude))
        sin_b = np.random.uniform(-sin_b_max, sin_b_max, n_directions)
        b = np.degrees(np.arcsin(sin_b))
    else:
        # Uniform over full sky
        cos_b = np.random.uniform(-1, 1, n_directions)
        b = np.degrees(np.arcsin(cos_b))
    
    l = np.random.uniform(0, 20, n_directions)
    return l, b

def compute_mean_probability(g_agamma, m_a_GeV, nu_GeV, n_directions=100, domain_size_kpc=0.01, ne_model="ne2001"):
    """
    Compute mean conversion probability averaged over multiple sky directions.
    
    Parameters:
    -----------
    g_agamma : float
        Axion-photon coupling [GeV^-1]
    m_a_GeV : float
        Axion mass [GeV]
    nu_GeV : float
        Photon frequency [GeV]
    n_directions : int
        Number of random sky directions to sample
    domain_size_kpc : float
        Coherence domain size [kpc]
    ne_model : str
        Electron density model ('ne2001', 'ymw16', etc.)
    
    Returns:
    --------
    float : Mean conversion probability
    """
    l_array, b_array = galactic_directions(n_directions)
    probabilities = []
    failed_count = 0
    
    for l, b in zip(l_array, b_array):
        try:
            P = P_gamma_to_alp(
                g_agamma=g_agamma,
                m_a=m_a_GeV,
                nu=nu_GeV,
                b=b,
                l=l,
                domain_size_kpc=domain_size_kpc,
                ne_model=ne_model
            )
            # Check for valid probability
            if not np.isnan(P) and not np.isinf(P) and P >= 0:
                probabilities.append(P)
            else:
                failed_count += 1
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            # Catch specific errors from pygedm or mixing calculations
            failed_count += 1
            continue
        except Exception as e:
            # Catch any other unexpected errors
            failed_count += 1
            continue
    
    if len(probabilities) == 0:
        print(f"Warning: All {n_directions} directions failed for m_a={m_a_GeV:.2e} GeV, nu={nu_GeV:.2e} GeV")
        return 0.0
    
    if failed_count > 0.5 * n_directions:
        print(f"Warning: {failed_count}/{n_directions} directions failed for m_a={m_a_GeV:.2e} GeV")
    
    return np.mean(probabilities)


def spectral_distortion(P_conversion, nu_hz):
    """
    Compute fractional spectral distortion due to axion-photon conversion.
    
    For small conversion probabilities:
    delta_f / f_gamma ≈ P_conversion
    
    This assumes the conversion happens along the line of sight and
    depletes the CMB photon energy at frequency nu.
    """
    return P_conversion

def compute_g_agamma_limit(m_a_eV, tms_freq_ghz, tms_sensitivity_ratio, 
                           g_base=1e-10, n_directions=50, domain_size_kpc=0.01):
    """
    Compute the limiting axion-photon coupling for a given axion mass.
    
    Strategy:
    1. Compute mean conversion probability at fiducial g_agamma
    2. Scale to match TMS sensitivity threshold
    
    Parameters:
    -----------
    m_a_eV : float
        Axion mass [eV]
    tms_freq_ghz : array
        TMS frequencies [GHz]
    tms_sensitivity_ratio : array
        TMS sensitivity delta_f / f_gamma
    g_base : float
        Fiducial coupling for initial calculation [GeV^-1]
    n_directions : int
        Number of sky directions to average
    
    Returns:
    --------
    float : Limiting g_agamma [GeV^-1]
    """
    try:
        m_a_GeV = m_a_eV * eV_to_GeV
        
        # Convert TMS frequencies to GeV
        nu_GeV_array = GHz_to_GeV(tms_freq_ghz)
        
        # Compute mean probability for each frequency
        P_mean_array = []
        for nu_GeV in nu_GeV_array:
            P_mean = compute_mean_probability(
                g_agamma=g_base,
                m_a_GeV=m_a_GeV,
                nu_GeV=nu_GeV,
                n_directions=n_directions,
                domain_size_kpc=domain_size_kpc
            )
            P_mean_array.append(P_mean)
        
        P_mean_array = np.array(P_mean_array)
        
        # Check if all probabilities are zero or too small
        if np.all(P_mean_array <= 0) or np.all(np.isnan(P_mean_array)):
            return np.nan
        
        # The spectral distortion scales as g_agamma^2 (since P ∝ g^2)
        # Find where P_mean(g_base) * (g/g_base)^2 = tms_sensitivity_ratio
        # => g = g_base * sqrt(tms_sensitivity_ratio / P_mean)
        
        valid = P_mean_array > 0
        if not np.any(valid):
            return np.nan
        
        g_candidates = g_base * np.sqrt(tms_sensitivity_ratio[valid] / P_mean_array[valid])
        
        # Return the most stringent (minimum) limit
        return np.nanmin(g_candidates)
        
    except Exception as e:
        print(f"Error computing limit for m_a = {m_a_eV:.2e} eV: {e}")
        return np.nan

def main():
    # Load TMS sensitivity data
    dat_path = str(REPO_ROOT / "TMSSensitivity.dat")
    tms_data = np.loadtxt(dat_path)
    tms_freq_ghz = tms_data[:, 0]  # GHz
    tms_delta_f = tms_data[:, 1]   # W m^-2 Hz^-1 sr^-1
    
    # CMB blackbody at TMS frequencies
    tms_nu_hz = tms_freq_ghz * 1e9
    tms_f_gamma = (2.0 * h_SI * tms_nu_hz**3 / c_SI**2) / np.expm1(h_SI * tms_nu_hz / (kB_SI * T_CMB_SI))
    
    # TMS fractional sensitivity
    tms_sensitivity_ratio = tms_delta_f / tms_f_gamma
    
    print(f"TMS frequency range: {tms_freq_ghz.min():.1f} – {tms_freq_ghz.max():.1f} GHz")
    print(f"TMS sensitivity ratio: {tms_sensitivity_ratio.min():.2e} – {tms_sensitivity_ratio.max():.2e}")
    
    # Axion mass range (eV)
    m_a_eV_array = np.logspace(-18, -10, 30)  # Reduced for testing
    
    # Fiducial coupling
    g_base = 1e-10  # GeV^-1
    
    # Number of sky directions - focus on galactic plane where models work best
    n_directions = 2
    
    # Try different models - ne2001 is usually most reliable
    ne_model = "ne2001"  # Can also try "ymw16"
    
    print(f"\nUsing electron density model: {ne_model}")
    print(f"Computing sensitivity for {len(m_a_eV_array)} mass points...")
    print(f"Averaging over {n_directions} sky directions per mass point")
    print(f"Restricting to galactic plane (|b| < 20°) for model reliability\n")
    
    # Parallel computation
    n_cores = 6
    # n_cores = max(1, cpu_count() - 1)
    print(f"Using {n_cores} CPU cores\n")
    
    worker = partial(
        compute_g_agamma_limit,
        tms_freq_ghz=tms_freq_ghz,
        tms_sensitivity_ratio=tms_sensitivity_ratio,
        g_base=g_base,
        n_directions=n_directions
    )
    
    with Pool(n_cores) as pool:
        g_lim_array = np.array(
            list(tqdm(pool.imap(worker, m_a_eV_array), total=len(m_a_eV_array), 
                     desc="Scanning masses"))
        )
    
    # Filter out NaN and unphysical values
    g_lim_array[g_lim_array > 1e-4] = np.nan  # Remove unphysically large couplings
    
    valid_points = np.sum(~np.isnan(g_lim_array))
    print(f"\nComputed limits for {valid_points} / {len(m_a_eV_array)} mass points")
    
    # Save results
    output_dir = REPO_ROOT / "AxionOscillations" / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "TMS_axion_sensitivity.csv"
    
    results = np.column_stack([m_a_eV_array, g_lim_array])
    np.savetxt(
        output_file,
        results,
        delimiter=",",
        header="m_a[eV],g_agamma[GeV^-1]",
        comments=""
    )
    print(f"Results saved to {output_file}")
    
    # Plot
    plot_sensitivity(m_a_eV_array, g_lim_array)

def plot_sensitivity(m_a_eV_array, g_lim_array):
    """
    Plot TMS sensitivity to axion-photon coupling.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    mask = ~np.isnan(g_lim_array)
    
    ax.loglog(m_a_eV_array[mask], g_lim_array[mask], 
              c='tab:blue', lw=2.5, label='TMS (projection)')
    ax.fill_between(m_a_eV_array[mask], g_lim_array[mask], 1e-6,
                    alpha=0.15, color='tab:blue')
    
    # Add existing constraints (you can add more here)
    # Example: CAST constraint
    # cast_data = np.loadtxt("path/to/CAST_data.txt")
    # ax.plot(cast_data[:,0], cast_data[:,1], label='CAST', ...)
    
    ax.set_xlabel(r"$m_a$ [eV]")
    ax.set_ylabel(r"$g_{a\gamma}$ [GeV$^{-1}$]")
    ax.set_title(r"TMS Sensitivity to Axion-Photon Coupling")
    
    ax.set_xlim(1e-9, 1e-3)
    ax.set_ylim(1e-15, 1e-6)
    
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = REPO_ROOT / "AxionOscillations" / "Figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "TMS_AxionSensitivity.pdf", bbox_inches='tight')
    print(f"Figure saved to {fig_dir / 'TMS_AxionSensitivity.pdf'}")
    
    plt.show()

if __name__ == "__main__":
    main()