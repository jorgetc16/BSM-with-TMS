#!/usr/bin/env python3
import numpy as np

# Physical constants (SI)
h = 6.62607015e-34       # Planck constant, J*s
kB = 1.380649e-23        # Boltzmann constant, J/K
c = 299792458.0          # speed of light, m/s

T_CMB = 2.7255           # CMB temperature in K

def BlackBody(nu_hz, T):
    """
    Planck law for specific intensity I_nu (W m^-2 Hz^-1 sr^-1).
    """
    x = h * nu_hz / (kB * T)
    # I_nu = 2 h nu^3 / c^2 * 1/(exp(x) - 1)
    return (2.0 * h * nu_hz**3 / c**2) / (np.expm1(x))

def compute_sensitivity_ratio(dat_path):
    # data: frequency (GHz) and delta f_gamma (W m^-2 Hz^-1 sr^-1)
    data = np.loadtxt(dat_path)
    freq_ghz = data[:, 0]
    delta_f = data[:, 1]

    nu_hz = freq_ghz * 1e9
    f_gamma = BlackBody(nu_hz, T_CMB)

    ratio = delta_f / f_gamma
    return freq_ghz, delta_f, f_gamma, ratio

def sensitivity_interpolator(dat_path):
    """
    Returns a function that interpolates sensitivity ratio delta_f/f_gamma
    for frequencies between 10 and 20 GHz.
    """
    freq_ghz, _, _, ratio = compute_sensitivity_ratio(dat_path)

    def interp_fn(freq_query_ghz):
        freq_query_ghz = np.asarray(freq_query_ghz)
        if np.any(freq_query_ghz < freq_ghz.min()) or np.any(freq_query_ghz > freq_ghz.max()):
            raise ValueError("Frequency out of bounds (10–20 GHz).")
        return np.interp(freq_query_ghz, freq_ghz, ratio)

    return interp_fn

if __name__ == "__main__":
    dat_path = "/home/jortecal/GitHub/TMS/TMSSensitivity.dat"
    freq_ghz, delta_f, f_gamma, ratio = compute_sensitivity_ratio(dat_path)

    print("# freq_GHz  delta_f_gamma  f_gamma_CMB  sensitivity_ratio")
    for f, df, fg, r in zip(freq_ghz, delta_f, f_gamma, ratio):
        print(f"{f:8.3f}  {df: .6e}  {fg: .6e}  {r: .6e}")

    # Example: interpolate at 12.5 GHz
    interp = sensitivity_interpolator(dat_path)
    print("Sensitivity ratio at 12.5 GHz:", interp(12.5))