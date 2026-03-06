from plasma import GHz_to_GeV, GeV_to_GHz
from mixing import P_gamma_to_alp
import numpy as np
import matplotlib.pyplot as plt

# P = P_gamma_to_alp(
#     g_agamma=1e-10,
#     m_a=1e-21,
#     nu=GHz_to_GeV(10.0),
#     b=0,
#     l=0, 
#     domain_size_kpc=0.5,
#     ne_model="alt"
# )
# print("Probability of conversion: ")
# print(P)

# Base parameters
base_params = {
    "g_agamma": 1e-4,
    "m_a": 1e-24,
    "nu": GHz_to_GeV(10.0),
    "b": 0,
    "l": 0,
    "domain_size_kpc": 0.01,
}

# Ne models to plot
ne_models = ["constant", "expsech", "ymw16", "ne2001"]
colors = ['blue', 'orange', 'green', 'red']

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Probability vs Mass
masses = np.logspace(-26, -17, 500)
for ne_model, color in zip(ne_models, colors):
    P_mass = [P_gamma_to_alp(**{**base_params, "m_a": m, "ne_model": ne_model}) for m in masses]
    axes[0, 0].loglog(masses*1e9, P_mass, '-', label=ne_model, color=color)
axes[0, 0].set_xlabel("Mass $m_a$ (eV)")
axes[0, 0].set_ylabel("Conversion Probability")
axes[0, 0].set_title(f"Probability vs Mass\n$g_{{a\\gamma}}$={base_params['g_agamma']:.1e}, $\\nu$={GeV_to_GHz(base_params['nu'])} GHz, $L$={base_params['domain_size_kpc']} kpc")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Probability vs Coupling
couplings = np.logspace(-14, -8, 20)
for ne_model, color in zip(ne_models, colors):
    P_coupling = [P_gamma_to_alp(**{**base_params, "g_agamma": g, "ne_model": ne_model}) for g in couplings]
    axes[0, 1].loglog(couplings, P_coupling, '-', label=ne_model, color=color)
axes[0, 1].set_xlabel("Coupling $g_{a\\gamma}$ (GeV$^{-1}$)")
axes[0, 1].set_ylabel("Conversion Probability")
axes[0, 1].set_title(f"Probability vs Coupling\n$m_a$={base_params['m_a']*1e9:.1e} eV, $\\nu$={GeV_to_GHz(base_params['nu'])} GHz, $L$={base_params['domain_size_kpc']} kpc")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, which='both')

# 3. Probability vs Frequency
frequencies_GHz = np.linspace(10, 20, 50)
frequencies_GeV = [GHz_to_GeV(f) for f in frequencies_GHz]
for ne_model, color in zip(ne_models, colors):
    P_freq = [P_gamma_to_alp(**{**base_params, "nu": f, "ne_model": ne_model}) for f in frequencies_GeV]
    axes[1, 0].semilogy(frequencies_GHz, P_freq, '-', label=ne_model, color=color)
axes[1, 0].set_xlabel("Frequency (GHz)")
axes[1, 0].set_ylabel("Conversion Probability")
axes[1, 0].set_title(f"Probability vs Frequency\n$m_a$={base_params['m_a']*1e9:.1e} eV, $g_{{a\\gamma}}$={base_params['g_agamma']:.1e}, $L$={base_params['domain_size_kpc']} kpc")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Probability vs Domain Size
domain_sizes = np.logspace(-6, 1, 500)
for ne_model, color in zip(ne_models, colors):
    P_domain = [P_gamma_to_alp(**{**base_params, "domain_size_kpc": d, "ne_model": ne_model}) for d in domain_sizes]
    axes[1, 1].loglog(domain_sizes, P_domain, '-', label=ne_model, color=color)
axes[1, 1].set_xlabel("Domain Size (kpc)")
axes[1, 1].set_ylabel("Conversion Probability")
axes[1, 1].set_title(f"Probability vs Domain Size\n$m_a$={base_params['m_a']*1e9:.1e} eV, $g_{{a\\gamma}}$={base_params['g_agamma']:.1e}, $\\nu$={GeV_to_GHz(base_params['nu'])} GHz")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("probability_analysis.png", dpi=300)
plt.show()

# print("Plot saved as 'probability_analysis.png'")

# ========================================================================
# Second figure: Spatial / directional dependence
# ========================================================================

# --- Base parameters for spatial plots ---
spatial_params = {
    "g_agamma": 1e-4,
    "m_a": 1e-24,
    "nu": GHz_to_GeV(10.0),
    "domain_size_kpc": 0.01,
}

ne_models_spatial = ["ymw16", "ne2001"]
colors_spatial = ['green', 'red']

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# --- 1. Probability vs Galactic Latitude (at l=0°) ---
latitudes = np.linspace(-90, 90, 181)
for ne_model, color in zip(ne_models_spatial, colors_spatial):
    P_lat = []
    for b in latitudes:
        P = P_gamma_to_alp(
            **spatial_params,
            b=b,
            l=0,
            ne_model=ne_model
        )
        P_lat.append(P)
    axes2[0, 0].semilogy(latitudes, P_lat, '-', label=ne_model, color=color)

axes2[0, 0].set_xlabel("Galactic Latitude $b$ (deg)")
axes2[0, 0].set_ylabel("Conversion Probability")
axes2[0, 0].set_title(
    f"Probability vs Latitude ($l=0°$)\n"
    f"$m_a$={spatial_params['m_a']*1e9:.1e} eV, "
    f"$g_{{a\\gamma}}$={spatial_params['g_agamma']:.1e}"
)
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].axvline(0, color='gray', ls='--', alpha=0.5)

# --- 2. Probability vs Galactic Longitude (at b=0°) ---
longitudes = np.linspace(0, 360, 361)
for ne_model, color in zip(ne_models_spatial, colors_spatial):
    P_lon = []
    for l in longitudes:
        P = P_gamma_to_alp(
            **spatial_params,
            b=0,
            l=l,
            ne_model=ne_model
        )
        P_lon.append(P)
    axes2[0, 1].semilogy(longitudes, P_lon, '-', label=ne_model, color=color)

axes2[0, 1].set_xlabel("Galactic Longitude $l$ (deg)")
axes2[0, 1].set_ylabel("Conversion Probability")
axes2[0, 1].set_title(
    f"Probability vs Longitude ($b=0°$)\n"
    f"$m_a$={spatial_params['m_a']*1e9:.1e} eV, "
    f"$g_{{a\\gamma}}$={spatial_params['g_agamma']:.1e}"
)
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].axvline(0, color='gray', ls='--', alpha=0.5, label='GC')

# --- 3. Full-sky Mollweide map ---
n_l, n_b = 180, 90  # resolution
l_grid = np.linspace(0, 360, n_l)
b_grid = np.linspace(-90, 90, n_b)
L_grid, B_grid = np.meshgrid(l_grid, b_grid)

# Use ymw16 for the sky map
P_sky = np.zeros_like(L_grid)
for i in range(n_b):
    for j in range(n_l):
        P_sky[i, j] = P_gamma_to_alp(
            **spatial_params,
            b=B_grid[i, j],
            l=L_grid[i, j],
            ne_model="ymw16"
        )

# Convert to Mollweide coordinates: longitude must be in [-pi, pi]
l_rad = np.deg2rad(L_grid)
l_rad[l_rad > np.pi] -= 2 * np.pi  # shift [0, 2pi] -> [-pi, pi]
b_rad = np.deg2rad(B_grid)

ax_moll = fig2.add_subplot(2, 2, 3, projection='mollweide')
axes2[1, 0].remove()  # remove the regular subplot, replace with Mollweide

im = ax_moll.pcolormesh(
    l_rad, b_rad, np.log10(P_sky),
    cmap='inferno', shading='auto'
)
ax_moll.set_title(
    f"Sky map (YMW16)\n"
    f"$m_a$={spatial_params['m_a']*1e9:.1e} eV, "
    f"$g_{{a\\gamma}}$={spatial_params['g_agamma']:.1e}",
    fontsize=12
)
ax_moll.grid(True, alpha=0.3)
cb = fig2.colorbar(im, ax=ax_moll, orientation='horizontal', pad=0.1, shrink=0.8)
cb.set_label(r"$\log_{10}(P_{\gamma \to a})$")

# --- 4. Probability vs Frequency for different latitudes ---
freq_range_GHz = np.linspace(5, 25, 80)
freq_range_GeV = [GHz_to_GeV(f) for f in freq_range_GHz]
lat_samples = [0, 10, 30, 60, 90]
cmap = plt.cm.viridis
lat_colors = [cmap(i / (len(lat_samples) - 1)) for i in range(len(lat_samples))]

for b_val, lc in zip(lat_samples, lat_colors):
    P_freq_lat = []
    for f in freq_range_GeV:
        P = P_gamma_to_alp(
            g_agamma=spatial_params["g_agamma"],
            m_a=spatial_params["m_a"],
            nu=f,
            b=b_val,
            l=0,
            domain_size_kpc=spatial_params["domain_size_kpc"],
            ne_model="ymw16"
        )
        P_freq_lat.append(P)
    axes2[1, 1].semilogy(freq_range_GHz, P_freq_lat, '-', color=lc,
                          label=f"$b={b_val}°$")

axes2[1, 1].set_xlabel("Frequency (GHz)")
axes2[1, 1].set_ylabel("Conversion Probability")
axes2[1, 1].set_title(
    f"Probability vs Frequency (YMW16, $l=0°$)\n"
    f"$m_a$={spatial_params['m_a']*1e9:.1e} eV, "
    f"$g_{{a\\gamma}}$={spatial_params['g_agamma']:.1e}"
)
axes2[1, 1].legend(fontsize=10)
axes2[1, 1].grid(True, alpha=0.3)

# --- 3. Probability vs Latitude for different longitudes (YMW16) ---
lon_samples = [0, 45, 90, 180, 270]
cmap_lon = plt.cm.plasma
lon_colors = [cmap_lon(i / (len(lon_samples) - 1)) for i in range(len(lon_samples))]

latitudes_fine = np.linspace(-90, 90, 91)
for l_val, lc in zip(lon_samples, lon_colors):
    P_lat_lon = []
    for b in latitudes_fine:
        P = P_gamma_to_alp(
            **spatial_params,
            b=b,
            l=l_val,
            ne_model="ymw16"
        )
        P_lat_lon.append(P)
    axes2[1, 0].semilogy(latitudes_fine, P_lat_lon, '-', color=lc,
                          label=f"$l={l_val}°$")

axes2[1, 0].set_xlabel("Galactic Latitude $b$ (deg)")
axes2[1, 0].set_ylabel("Conversion Probability")
axes2[1, 0].set_title(
    f"Probability vs Latitude (YMW16)\n"
    f"$m_a$={spatial_params['m_a']*1e9:.1e} eV, "
    f"$g_{{a\\gamma}}$={spatial_params['g_agamma']:.1e}"
)
axes2[1, 0].legend(fontsize=10)
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].axvline(0, color='gray', ls='--', alpha=0.5)

fig2.tight_layout()
fig2.savefig("probability_spatial_analysis.png", dpi=300)
plt.show()