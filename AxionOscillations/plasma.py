# plasma.py
import numpy as np

try:
    import pygedm
except Exception:
    pygedm = None

# -----------------------
# Constants (natural units)
# -----------------------
alpha_em = 1 / 137
m_e_GeV = 0.000511  # electron mass

# -----------------------
# Unit conversions
# -----------------------
def GHz_to_GeV(GHz):
    return 4.135667696e-15 * GHz

def GeV_to_GHz(GeV):
    return GeV / 4.135667696e-15
    
def microGauss_to_GeV2(B_uG):
    return B_uG * 1.95e-26

def kpc_to_inverseGeV(kpc):
    return kpc * 3.085677581e19 / 1.973269804e-16

# -----------------------
# Mixing terms
# -----------------------
def Delta_a(m_a_GeV, nu_GeV):
    return -m_a_GeV**2 / (2.0 * nu_GeV)

def omega_plasma_sq(n_e_GeV3):
    return 4.0 * np.pi * alpha_em * n_e_GeV3 / m_e_GeV

def Delta_plasma(n_e_GeV3, nu_GeV):
    return -omega_plasma_sq(n_e_GeV3) / (2.0 * nu_GeV)

# -----------------------
# Electron density models
# -----------------------
N_0 = 3.2e-4
r_ne_0 = 5.0
z_ne_0 = 1.0

N_1 = 0.035
A_1 = 17.0
H   = 1.8
R_sun = -8.0

# Constant density (already converted)
n_e_constant_GeV3 = 7.68e-45

def ne_expsech(r, z):
    return N_0 * np.exp(-r / r_ne_0) * (1.0 / np.cosh(z / z_ne_0))**2

def ne_alt(r, z):
    val = N_1 * np.cos(np.pi * r / (2 * A_1)) / np.cos(np.pi * R_sun / (2 * A_1))
    val *= (1.0 / np.cosh(z / H))**2
    val *= (1.0 if (r - A_1) > 0 else 0.0)
    return val

def _to_deg(angle):
    # Heuristic: if |angle| <= 2π, assume radians
    if np.abs(angle) <= 2.0 * np.pi:
        return np.degrees(angle)
    return angle

def electron_density_at_GeV3(d_kpc, b, l, Galactic_to_Cylindrical, model="constant"):
    if model == "constant":
        return n_e_constant_GeV3

    cm3_to_GeV3 = (5.0677307e13)**-3

    if model in {"ymw16", "ne2001", "pygedm_ymw16", "pygedm_ne2001"}:
        if pygedm is None:
            raise ImportError("pygedm is not installed. Please install pygedm to use this model.")
        gl_deg = _to_deg(l)
        gb_deg = _to_deg(b)
        dist_pc = d_kpc * 1000.0
        method = "ymw16" if model in {"ymw16", "pygedm_ymw16"} else "ne2001"
        ne_cm3 = pygedm.calculate_electron_density_lbr(gl_deg, gb_deg, dist_pc, method=method)
        return float(ne_cm3.to_value()) * cm3_to_GeV3

    r, z, _ = Galactic_to_Cylindrical(d_kpc, b, l)

    if model == "expsech":
        return ne_expsech(r, z) * cm3_to_GeV3
    if model == "alt":
        return ne_alt(r, z) * cm3_to_GeV3

    return n_e_constant_GeV3
