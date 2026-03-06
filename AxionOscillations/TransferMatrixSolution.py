from plasma import GHz_to_GeV, GeV_to_GHz
from mixing import P_gamma_to_alp
import numpy as np
import matplotlib.pyplot as plt

P = P_gamma_to_alp(
    g_agamma=1e-10,
    m_a=1e-21,
    nu=GHz_to_GeV(10.0),
    b=0,
    l=0, 
    domain_size_kpc=0.01,
    ne_model="ae200t"
)
print("Probability of conversion: ")
print(P)
