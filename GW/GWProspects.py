#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys
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
# Ensure repo root is on path to import TMSSensitivity.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from TMSSensitivity import sensitivity_interpolator  # noqa: E402

# Physical constants (SI)
c = 299792458.0                # m/s
kB = 1.380649e-23              # J/K
hbar = 1.054571817e-34         # J*s
G = 6.67430e-11                # m^3 kg^-1 s^-2

T_CMB = 2.7255                 # K

# X_e(z) reference points
_Z_POINTS = np.array([0.0, 10.0, 20.0, 1100.0])
_XE_POINTS = np.array([1.0, 0.68, 0.0002, 0.15])

def ionization_fraction(z):
    """
    Linear interpolation of X_e(z) using the provided reference points.
    Valid for 0 <= z <= 1100.
    """
    z = np.asarray(z)
    if np.any(z < _Z_POINTS.min()) or np.any(z > _Z_POINTS.max()):
        raise ValueError("z is out of bounds (0–1100).")
    return np.interp(z, _Z_POINTS, _XE_POINTS)

def compute_I_zini(z_ini, n_steps=5000):
    """
    I(z_ini) = ∫_0^{z_ini} dz (1+z)^(-3/2) X_e(z)^(-2)
    """
    z = np.linspace(0.0, z_ini, n_steps)
    xe = ionization_fraction(z)
    integrand = (1.0 + z)**(-1.5) / (xe**2)
    return np.trapezoid(integrand, z)

def critical_density(H0_SI):
    return 3.0 * H0_SI**2 / (8.0 * np.pi * G)

def omega_gamma(T_K, H0_SI):
    """
    Photon energy density parameter Ω_γ.
    ρ_γ = (π^2/15) * (kB T)^4 / (ħ^3 c^5)
    """
    rho_gamma = (np.pi**2 / 15.0) * (kB * T_K)**4 / (hbar**3 * c**5)
    return rho_gamma / critical_density(H0_SI)

def conversion_probability_intf(f_hz, B0_nG=1.0, delta_z0_Mpc=1.0, I_zini=1e6):
    """
    P ≈ 6.3e-19 * (B0/nG)^2 * (ω0/T0)^2 * (Mpc/Δz0) * (I/1e6)

    Using T0/(2π) = 56.78 GHz ⇒ (ω0/T0) = f / (56.78 GHz).
    """
    f_ref = 56.78e9  # Hz
    omega_over_T = f_hz / f_ref
    return 6.3e-19 * (B0_nG**2) * (omega_over_T**2) * (1.0 / delta_z0_Mpc) * (I_zini / 1e6)

def omega_gw_from_sensitivity(delta_f_over_f, f_hz, intf, H0_SI):
    """
    From δf/f = (π^4/15) (T/ω)^3 * P * (Ω_GW/Ω_γ).
    """
    T_ang = kB * T_CMB / hbar     # angular-frequency units (rad/s)
    omega = 2.0 * np.pi * f_hz    # rad/s

    pref = (15.0 / np.pi**4) * (omega / T_ang)**3
    return pref * (delta_f_over_f / intf) * omega_gamma(T_CMB, H0_SI)

def hc_from_omega_gw(omega_gw, f_hz, H0_SI):
    return np.sqrt((3.0 * H0_SI**2 / (4.0 * np.pi**2)) * omega_gw * f_hz**-2)

def prospects_hc(
    f_ghz,
    dat_path,
    B0_nG=1,
    delta_z0_Mpc=1.0,
    I_zini=1e6,
    z_ini=None,
    H0_km_s_Mpc=67.4
):
    """
    Compute h_c(f) using TMS sensitivity and the conversion probability.
    If z_ini is provided, I_zini is computed from it.
    """
    f_hz = np.asarray(f_ghz) * 1e9
    H0_SI = H0_km_s_Mpc * 1000.0 / (3.085677581e22)  # s^-1

    interp = sensitivity_interpolator(dat_path)
    delta_f_over_f = interp(np.asarray(f_ghz))

    if z_ini is not None:
        I_zini = compute_I_zini(z_ini)

    intf = conversion_probability_intf(
        f_hz, B0_nG=B0_nG, delta_z0_Mpc=delta_z0_Mpc, I_zini=I_zini
    )

    omega_gw = omega_gw_from_sensitivity(delta_f_over_f, f_hz, intf, H0_SI)
    hc = hc_from_omega_gw(omega_gw, f_hz, H0_SI)

    return omega_gw, hc

if __name__ == "__main__":
    dat_path = str(REPO_ROOT / "TMSSensitivity.dat")

    # Dense sampling for smooth curve
    f_ghz = np.linspace(10.0, 20.0, 400)
    f_hz = f_ghz * 1e9

    # Base curve (computed using z_ini)
    omega_gw_base, hc_base = prospects_hc(f_ghz, dat_path, z_ini=1100)

    # Contour values for (T0/ω0)^2 * P
    contour_values = [1e-37, 1e-34, 1e-31, 1e-28, 1e-25, 1e-22, 1e-19]

    # Plot (log-log, frequency in Hz)

    fig = plt.figure(figsize=(8, 7))

    delta_z0_Mpc = 1.0
    z_ini = 1100
    I_zini = compute_I_zini(z_ini)

    base_pref = 6.3e-19 * (1.0 / delta_z0_Mpc) * (I_zini / 1e6)

    for i, s in enumerate(contour_values):
        B0_nG = np.sqrt(s / base_pref)
        omega_gw, hc = prospects_hc(
            f_ghz,
            dat_path,
            B0_nG=B0_nG,
            delta_z0_Mpc=delta_z0_Mpc,
            z_ini=z_ini
        )
        line, = plt.loglog(f_hz, hc)

        # Place label at the midpoint of each curve
        mid = len(f_hz) // 2
        # Format exponent nicely
        exp = int(np.log10(s))
        label_text = fr"$\left(T_0/\omega_0\right)^2\mathcal{{P}}=10^{{{exp}}}$"

        plt.annotate(
            label_text,
            xy=(f_hz[mid], hc[mid]),
            fontsize=11,
            color=line.get_color(),
            ha='center', va='bottom',
            rotation=-2,
            path_effects=[
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ],
        )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Characteristic strain $h_c$")
    plt.grid(True, which="major", alpha=0.3)
    plt.xlim(8e9, 2.5e10)

    plt.tight_layout()
    plt.savefig("/home/jortecal/GitHub/TMS/GW/Figures/GWProspects.pdf")
    # plt.show()