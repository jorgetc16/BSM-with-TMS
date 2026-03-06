# mixing.py
import numpy as np
from scipy.linalg import expm

from plasma import (
    Delta_a,
    Delta_plasma,
    microGauss_to_GeV2,
    kpc_to_inverseGeV,
    electron_density_at_GeV3
)
from gmf import B_transverse, Galactic_to_Cylindrical

# -----------------------
# Hamiltonian
# -----------------------
def mixing_hamiltonian(g_agamma, m_a, nu, B_perp_uG, n_e_GeV3):
    Delta_pl = Delta_plasma(n_e_GeV3, nu)
    Delta_a_val = Delta_a(m_a, nu)
    Delta_ag = 0.5 * g_agamma * microGauss_to_GeV2(B_perp_uG)

    H = np.zeros((3, 3), dtype=np.complex128)
    H[0, 0] = Delta_pl
    H[1, 1] = Delta_pl
    H[2, 2] = Delta_a_val
    H[1, 2] = Delta_ag
    H[2, 1] = Delta_ag

    return H

# -----------------------
# Transfer matrix
# -----------------------
def transfer_matrix(H, L_kpc):
    L = kpc_to_inverseGeV(L_kpc)
    return expm(-1j * H * L)

# -----------------------
# Full propagation
# -----------------------
def P_gamma_to_alp(
    g_agamma,
    m_a,
    nu,
    b,
    l,
    d_max_kpc=200.0,
    domain_size_kpc=0.5,
    ne_model="constant"
):
    rho = np.zeros((3, 3), dtype=np.complex128)
    rho[0, 0] = rho[1, 1] = 0.5

    N = int(np.ceil(d_max_kpc / domain_size_kpc))

    for i in range(N):
        d = (i + 0.5) * domain_size_kpc
        if d > d_max_kpc:
            break

        Bb, Bl = B_transverse(d, b, l)
        B_perp = np.sqrt(Bb**2 + Bl**2)
        if B_perp == 0:
            continue

        n_e = electron_density_at_GeV3(
            d, b, l,
            Galactic_to_Cylindrical,
            model=ne_model
        )

        H = mixing_hamiltonian(g_agamma, m_a, nu, B_perp, n_e)
        T = transfer_matrix(H, domain_size_kpc)
        rho = T @ rho @ T.conj().T

    return np.real(rho[2, 2])
