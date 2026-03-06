import numpy as np

def galactic_directions(n_directions=100, avoid_poles=True, max_latitude=20):
    """
    Generate uniformly distributed directions across the sky.
    Returns (l, b) in degrees, where l is galactic longitude, b is latitude.
    
    Parameters:
    -----------
    n_directions : int
        Number of directions to generate
    avoid_poles : bool
        If True, restrict to galactic plane region
    max_latitude : float
        Maximum |b| in degrees (only used if avoid_poles=True)
    """
    if avoid_poles:
        # Sample uniformly in galactic plane region where models are most reliable
        # Uniform in sin(b) for uniform coverage
        sin_b_max = np.sin(np.radians(max_latitude))
        sin_b = np.random.uniform(-sin_b_max, sin_b_max, n_directions)
        b = np.degrees(np.arcsin(sin_b))
    else:
        # Uniform over full sky
        cos_b = np.random.uniform(-1, 1, n_directions)
        b = np.degrees(np.arcsin(cos_b))
    
    l = np.random.uniform(0, 360, n_directions)
    return l, b

def compute_mean_probability(g_agamma, m_a_GeV, nu_GeV, n_directions=100, domain_size_kpc=0.01, ne_model="ne2001"):
    """
    Compute mean conversion probability averaged over multiple sky directions.
    
    Parameters:
    -----------
    g_agamma : float
        Axion-photon coupling [GeV^-1]
    m_a_GeV : float
        Axion mass [GeV]
    nu_GeV : float
        Photon frequency [GeV]
    n_directions : int
        Number of random sky directions to sample
    domain_size_kpc : float
        Coherence domain size [kpc]
    ne_model : str
        Electron density model ('ne2001', 'ymw16', etc.)
    
    Returns:
    --------
    float : Mean conversion probability
    """
    l_array, b_array = galactic_directions(n_directions)
    probabilities = []
    failed_count = 0
    
    for l, b in zip(l_array, b_array):
        try:
            P = P_gamma_to_alp(
                g_agamma=g_agamma,
                m_a=m_a_GeV,
                nu=nu_GeV,
                b=b,
                l=l,
                domain_size_kpc=domain_size_kpc,
                ne_model=ne_model
            )
            # Check for valid probability
            if not np.isnan(P) and not np.isinf(P) and P >= 0:
                probabilities.append(P)
            else:
                failed_count += 1
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            # Catch specific errors from pygedm or mixing calculations
            failed_count += 1
            continue
        except Exception as e:
            # Catch any other unexpected errors
            failed_count += 1
            continue
    
    if len(probabilities) == 0:
        print(f"Warning: All {n_directions} directions failed for m_a={m_a_GeV:.2e} GeV, nu={nu_GeV:.2e} GeV")
        return 0.0
    
    if failed_count > 0.5 * n_directions:
        print(f"Warning: {failed_count}/{n_directions} directions failed for m_a={m_a_GeV:.2e} GeV")
    
    return np.mean(probabilities)