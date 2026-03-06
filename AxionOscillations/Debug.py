from plasma import GHz_to_GeV
from mixing import P_gamma_to_alp

P = P_gamma_to_alp(
    g_agamma=1e-10,
    m_a=1e-21,
    nu=GHz_to_GeV(10.0),
    b=0.0,
    l=0.0,
    domain_size_kpc=0.01,
    ne_model="ne2001",
)
print(P)