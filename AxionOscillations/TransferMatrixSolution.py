from plasma import GHz_to_GeV, GeV_to_GHz
from mixing import P_gamma_to_alp
import numpy as np
import matplotlib.pyplot as plt

P = P_gamma_to_alp(
    g_agamma=1e-10,
    m_a=1.487352107293511852e-11*1e-9,
    nu=GHz_to_GeV(15.0),
    b=5,
    l=5, 
    domain_size_kpc=0.01,
    ne_model="ne2001"
)

print("Probability of conversion: ")
print(P)
limitTMS = 1e-10 * np.sqrt(9e-7/ P)
print(f"Estimated TMS sensitivity limit: g_agamma < {limitTMS:.2e} GeV^-1")