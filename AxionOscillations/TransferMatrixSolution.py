from plasma import GHz_to_GeV, GeV_to_GHz
from mixing import P_gamma_to_alp
import numpy as np
import matplotlib.pyplot as plt

P = P_gamma_to_alp(
    g_agamma=1e-10,
    m_a=1e-26,
    nu=GHz_to_GeV(10.0),
    b=40,
    l=50, 
    domain_size_kpc=0.01,
    ne_model="ne2001"
)
print("Probability of conversion: ")
print(P)
