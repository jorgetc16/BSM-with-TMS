import sys
import numpy as np
import healpy as hp
from plasma import GHz_to_GeV
from mixing import P_gamma_to_alp

def calculate_pixel_probability(ipix, nside, nu, g_agamma, m_a, d_max_kpc=300.0, domain_size_kpc=0.01, ne_model="constant"):
    """Calculate conversion probability for a single pixel."""
    theta, phi = hp.pix2ang(nside, ipix)
    b = np.pi/2 - theta
    l = phi
    
    try:
        P = P_gamma_to_alp(
            g_agamma=g_agamma,
            m_a=m_a,
            nu=nu,
            b=b,
            l=l,
            d_max_kpc=d_max_kpc,
            domain_size_kpc=domain_size_kpc,
            ne_model=ne_model
        )
        return (ipix, P)
    except Exception as e:
        print(f"Warning: Error at pixel {ipix}: {e}", file=sys.stderr)
        return (ipix, 0.0)

if __name__ == "__main__":
    # Parse command line arguments
    nside = int(sys.argv[1])
    ipix = int(sys.argv[2])
    g_agamma = float(sys.argv[3])
    m_a = float(sys.argv[4])*1e-9  # Convert eV to GeV
    nu_GHz = float(sys.argv[5])
    ne_model = sys.argv[6]
    domain_size_kpc = float(sys.argv[7])
    
    nu = GHz_to_GeV(nu_GHz)
    
    # Calculate probability for this pixel
    pixel_idx, probability = calculate_pixel_probability(
        ipix, nside, nu, g_agamma, m_a,
        d_max_kpc=300.0,
        domain_size_kpc=domain_size_kpc,
        ne_model=ne_model
    )
    
    # Output result (to be collected by master process)
    print(f"{pixel_idx} {probability}")