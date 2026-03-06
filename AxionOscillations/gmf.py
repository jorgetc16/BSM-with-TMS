# gmf.py
import numpy as np

# -----------------------
# Coordinates
# -----------------------
def Galactic_to_Cylindrical(d_kpc, b, l):
    R_Earth = -8.5
    r = np.sqrt(d_kpc**2 * np.cos(b)**2 +
                R_Earth**2 +
                2 * R_Earth * d_kpc * np.cos(b) * np.cos(l))
    z = d_kpc * np.sin(b)
    phi = np.arctan2(
        d_kpc * np.cos(b) * np.sin(l),
        d_kpc * np.cos(b) * np.cos(l) + R_Earth
    )
    return r, z, phi

# -----------------------
# JF-like model parameters
# -----------------------
z0 = 5.3
r_n = 9.22
r_s = 16.7
w_h = 0.2
h_disk = 0.4
w_disk = 0.27

B_n = 1.4
B_s = -1.1
B_X = 4.6
r_cX = 4.8
Theta_X_0 = 0.86
r_x = 2.9

def L_function(l, h, w):
    return (1 + np.exp(-2 * (np.abs(l) - h) / w))**(-1)

# -----------------------
# Magnetic field components
# -----------------------
def B_Disk(r, z):
    return 0.0, 0.0, 0.0

def B_Toroidal(r, z):
    if np.sqrt(r**2 + z**2) < 1.0 or r > 20.0:
        return 0.0, 0.0, 0.0

    if z >= 0:
        Bphi = np.exp(-np.abs(z)/z0) * L_function(z, h_disk, w_disk) * B_n * (1 - L_function(r, r_n, w_h))
    else:
        Bphi = np.exp(-np.abs(z)/z0) * L_function(z, h_disk, w_disk) * B_s * (1 - L_function(r, r_s, w_h))

    return 0.0, 0.0, Bphi

def B_Poloidal(r, z):
    if np.sqrt(r**2 + z**2) < 1.0 or r > 20.0 or r == 0:
        return 0.0, 0.0, 0.0

    bx = lambda x: B_X * np.exp(-x / r_x)
    r_p = r * r_cX / (r_cX + np.abs(z) / np.tan(Theta_X_0))
    r_p0 = r - np.abs(z) / np.tan(Theta_X_0)

    if r_p >= r_cX:
        Theta = Theta_X_0
        Br = bx(max(r_p0, 0.0)) * r_p0 / r * np.cos(Theta)
        Bz = bx(max(r_p0, 0.0)) * r_p0 / r * np.sin(Theta)
    else:
        Theta = np.arctan2(np.abs(z), r - r_p)
        Br = bx(r_p) * (r_p / r)**2 * np.cos(Theta)
        Bz = bx(r_p) * (r_p / r)**2 * np.sin(Theta)

    return Br, Bz, 0.0

def B_MW(r, z, phi):
    Br1, Bz1, Bp1 = B_Disk(r, z)
    Br2, Bz2, Bp2 = B_Toroidal(r, z)
    Br3, Bz3, Bp3 = B_Poloidal(r, z)
    return Br1 + Br2 + Br3, Bz1 + Bz2 + Bz3, Bp1 + Bp2 + Bp3

# -----------------------
# Transverse projection
# -----------------------
def B_transverse(d_kpc, b, l):
    r, z, phi = Galactic_to_Cylindrical(d_kpc, b, l)
    Br, Bz, Bphi = B_MW(r, z, phi)

    B_b = np.sin(b) * (Br*np.cos(l - phi) + Bphi*np.sin(l - phi)) - Bz*np.cos(b)
    B_l = Br*np.sin(phi - l) + Bphi*np.cos(phi - l)

    return B_b, B_l
