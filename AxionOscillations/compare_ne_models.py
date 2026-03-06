import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plasma import ne_expsech
from gmf import Galactic_to_Cylindrical

try:
    import pygedm
except Exception:
    print("pygedm not installed; ymw16 and ne2001 models will be unavailable.")
    pygedm = None

def _pygedm_ne_xyz(r_kpc, z_kpc, method):
    if pygedm is None:
        return np.nan
    x_pc = r_kpc * 1000.0
    y_pc = 0.0
    z_pc = z_kpc * 1000.0
    ne_cm3 = pygedm.calculate_electron_density_xyz(x_pc, y_pc, z_pc, method=method)
    return float(ne_cm3.to_value())

# ======================
# 1D Profiles
# ======================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Radial profile at z=0
r_values = np.linspace(0, 25, 500)
ne_expsech_r = [ne_expsech(r, 0) for r in r_values]

ne_ymw16_r = [_pygedm_ne_xyz(r, 0, "ymw16") for r in r_values] if pygedm else None
ne_ne2001_r = [_pygedm_ne_xyz(r, 0, "ne2001") for r in r_values] if pygedm else None

axes[0, 0].semilogy(r_values, ne_expsech_r, 'b-', label='expsech', linewidth=2)
if pygedm:
    axes[0, 0].semilogy(r_values, ne_ymw16_r, 'g--', label='ymw16 (pygedm)', linewidth=2)
    axes[0, 0].semilogy(r_values, ne_ne2001_r, 'm--', label='ne2001 (pygedm)', linewidth=2)

axes[0, 0].axvline(8.0, color='k', linestyle='--', alpha=0.5, label='Solar position')
axes[0, 0].set_xlabel('Radius r (kpc)', fontsize=12)
axes[0, 0].set_ylabel('Electron Density (cm$^{-3}$)', fontsize=12)
axes[0, 0].set_title('Radial Profile at z=0', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Vertical profile at r=8 kpc (Solar position)
z_values = np.linspace(-5, 5, 500)
ne_expsech_z = [ne_expsech(8.0, z) for z in z_values]

ne_ymw16_z = [_pygedm_ne_xyz(8.0, z, "ymw16") for z in z_values] if pygedm else None
ne_ne2001_z = [_pygedm_ne_xyz(8.0, z, "ne2001") for z in z_values] if pygedm else None

axes[0, 1].semilogy(z_values, ne_expsech_z, 'b-', label='expsech', linewidth=2)
if pygedm:
    axes[0, 1].semilogy(z_values, ne_ymw16_z, 'g--', label='ymw16 (pygedm)', linewidth=2)
    axes[0, 1].semilogy(z_values, ne_ne2001_z, 'm--', label='ne2001 (pygedm)', linewidth=2)

axes[0, 1].set_xlabel('Height z (kpc)', fontsize=12)
axes[0, 1].set_ylabel('Electron Density (cm$^{-3}$)', fontsize=12)
axes[0, 1].set_title('Vertical Profile at r=8 kpc', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Radial profile at different heights
z_heights = [0, 0.5, 1.0, 2.0]
colors_z = ['black', 'blue', 'green', 'orange']

for z_h, col in zip(z_heights, colors_z):
    ne_exp = [ne_expsech(r, z_h) for r in r_values]
    axes[1, 0].semilogy(r_values, ne_exp, '-', color=col, linewidth=1.5, 
                        label=f'expsech z={z_h} kpc')
    if pygedm:
        ne_ymw16 = [_pygedm_ne_xyz(r, z_h, "ymw16") for r in r_values]
        ne_ne2001 = [_pygedm_ne_xyz(r, z_h, "ne2001") for r in r_values]
        axes[1, 0].semilogy(r_values, ne_ymw16, '--', color=col, linewidth=1.5,
                            label=f'ymw16 z={z_h} kpc')
        axes[1, 0].semilogy(r_values, ne_ne2001, ':', color=col, linewidth=1.5,
                            label=f'ne2001 z={z_h} kpc')

axes[1, 0].set_xlabel('Radius r (kpc)', fontsize=12)
axes[1, 0].set_ylabel('Electron Density (cm$^{-3}$)', fontsize=12)
axes[1, 0].set_title('Radial Profiles at Different Heights', fontsize=13)
axes[1, 0].legend(fontsize=8, ncol=2)
axes[1, 0].grid(True, alpha=0.3)

# Remove the ratio plot - hide this subplot
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('ne_models_1D_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ne_models_1D_comparison.png")

# ======================
# 2D Maps
# ======================
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Create 2D grid
r_grid = np.linspace(0, 25, 200)
z_grid = np.linspace(-5, 5, 200)
R, Z = np.meshgrid(r_grid, z_grid)

# Calculate densities on grid
ne_expsech_grid = np.zeros_like(R)
ne_ymw16_grid = np.zeros_like(R)
ne_ne2001_grid = np.zeros_like(R)

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        r = R[i, j]
        z = Z[i, j]
        ne_expsech_grid[i, j] = ne_expsech(r, z)
        if pygedm:
            ne_ymw16_grid[i, j] = _pygedm_ne_xyz(r, z, "ymw16")
            ne_ne2001_grid[i, j] = _pygedm_ne_xyz(r, z, "ne2001")
        else:
            ne_ymw16_grid[i, j] = np.nan
            ne_ne2001_grid[i, j] = np.nan

# Log normalization (avoid zeros)
eps = 1e-12
norm_exp = LogNorm(vmin=max(np.nanmin(ne_expsech_grid), eps),
                   vmax=np.nanmax(ne_expsech_grid))

# expsech
im0 = axes[0].pcolormesh(R, Z, ne_expsech_grid, shading='auto', norm=norm_exp, cmap='viridis')
axes[0].set_xlabel('Radius r (kpc)', fontsize=12)
axes[0].set_ylabel('Height z (kpc)', fontsize=12)
axes[0].set_title('expsech', fontsize=13)
axes[0].plot(8.0, 0, 'r*', markersize=15, label='Sun')
axes[0].legend()
plt.colorbar(im0, ax=axes[0], label='$n_e$ (cm$^{-3}$)')

# ymw16 (if available)
if pygedm:
    norm_ymw16 = LogNorm(vmin=max(np.nanmin(ne_ymw16_grid), eps),
                         vmax=np.nanmax(ne_ymw16_grid))
    im1 = axes[1].pcolormesh(R, Z, ne_ymw16_grid, shading='auto', norm=norm_ymw16, cmap='viridis')
    axes[1].set_xlabel('Radius r (kpc)', fontsize=12)
    axes[1].set_ylabel('Height z (kpc)', fontsize=12)
    axes[1].set_title('ymw16 (pygedm)', fontsize=13)
    axes[1].plot(8.0, 0, 'r*', markersize=15, label='Sun')
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1], label='$n_e$ (cm$^{-3}$)')
else:
    axes[1].text(0.5, 0.5, "PyGEDM not installed", ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('ymw16 (pygedm)')

# ne2001 (if available)
if pygedm:
    norm_ne2001 = LogNorm(vmin=max(np.nanmin(ne_ne2001_grid), eps),
                          vmax=np.nanmax(ne_ne2001_grid))
    im2 = axes[2].pcolormesh(R, Z, ne_ne2001_grid, shading='auto', norm=norm_ne2001, cmap='viridis')
    axes[2].set_xlabel('Radius r (kpc)', fontsize=12)
    axes[2].set_ylabel('Height z (kpc)', fontsize=12)
    axes[2].set_title('ne2001 (pygedm)', fontsize=13)
    axes[2].plot(8.0, 0, 'r*', markersize=15, label='Sun')
    axes[2].legend()
    plt.colorbar(im2, ax=axes[2], label='$n_e$ (cm$^{-3}$)')
else:
    axes[2].text(0.5, 0.5, "PyGEDM not installed", ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('ne2001 (pygedm)')

plt.tight_layout()
plt.savefig('ne_models_2D_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ne_models_2D_comparison.png")

# ======================
# Line-of-Sight Integration
# ======================
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# Define sight lines: different galactic latitudes at l=0
b_values = np.linspace(-np.pi/2, np.pi/2, 7)
d_max = 20.0  # kpc
distances = np.linspace(0.01, d_max, 500)

for b in b_values:
    integral_expsech = []
    integral_ymw16 = []
    integral_ne2001 = []

    for d in distances:
        # Integrate from 0 to d
        d_steps = np.linspace(0.01, d, 100)
        ne_exp_sum = 0.0
        ne_ymw16_sum = 0.0
        ne_ne2001_sum = 0.0

        for d_i in d_steps:
            r, z, _ = Galactic_to_Cylindrical(d_i, b, 0)
            ne_exp_sum += ne_expsech(r, z)
            if pygedm:
                ne_ymw16_sum += _pygedm_ne_xyz(r, z, "ymw16")
                ne_ne2001_sum += _pygedm_ne_xyz(r, z, "ne2001")

        step = d_steps[1] - d_steps[0]
        integral_expsech.append(ne_exp_sum * step)
        if pygedm:
            integral_ymw16.append(ne_ymw16_sum * step)
            integral_ne2001.append(ne_ne2001_sum * step)

    b_deg = np.degrees(b)
    axes.plot(distances, integral_expsech, '-', label=f'expsech b={b_deg:.0f}°')
    if pygedm:
        axes.plot(distances, integral_ymw16, '--', label=f'ymw16 b={b_deg:.0f}°')
        axes.plot(distances, integral_ne2001, ':', label=f'ne2001 b={b_deg:.0f}°')

axes.set_xlabel('Distance (kpc)', fontsize=12)
axes.set_ylabel('Column Density (cm$^{-2}$)', fontsize=12)
axes.set_title('Line-of-Sight Integration', fontsize=13)
axes.set_yscale('log')
axes.legend(ncol=2, fontsize=8)
axes.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ne_models_LOS_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ne_models_LOS_comparison.png")

# ======================
# Statistics Summary
# ======================
print("\n" + "="*50)
print("ELECTRON DENSITY MODEL COMPARISON")
print("="*50)

# At solar position
ne_exp_sun = ne_expsech(8.0, 0)
print(f"\nAt Solar Position (r=8 kpc, z=0):")
print(f"  expsech: {ne_exp_sun:.4e} cm⁻³")

# At galactic center
ne_exp_gc = ne_expsech(0.1, 0)
print(f"\nNear Galactic Center (r=0.1 kpc, z=0):")
print(f"  expsech: {ne_exp_gc:.4e} cm⁻³")

# Scale heights
print(f"\nScale Parameters:")
print(f"  expsech: r₀={5.0} kpc, z₀={1.0} kpc")

if pygedm:
    ne_ymw16_sun = _pygedm_ne_xyz(8.0, 0, "ymw16")
    ne_ne2001_sun = _pygedm_ne_xyz(8.0, 0, "ne2001")
    print(f"\nAt Solar Position (comparison):")
    print(f"  ymw16:   {ne_ymw16_sun:.4e} cm⁻³")
    print(f"  ne2001:  {ne_ne2001_sun:.4e} cm⁻³")

plt.show()