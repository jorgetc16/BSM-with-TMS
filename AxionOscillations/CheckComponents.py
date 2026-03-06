import numpy as np
import matplotlib.pyplot as plt
from plasma import (
    GHz_to_GeV, GeV_to_GHz, microGauss_to_GeV2, 
    kpc_to_inverseGeV, alpha_em, m_e_GeV,
    omega_plasma_sq, Delta_a, Delta_plasma
)

def critical_mass_plasma(n_e_GeV3):
    """
    Critical mass where m_a^2/(2ν) ~ ω_p^2/(2ν)
    Returns: m_a,crit [GeV]
    """
    omega_p = np.sqrt(omega_plasma_sq(n_e_GeV3))
    return omega_p

def critical_mass_mixing(nu_GeV, g_agamma, B_perp_uG):
    """
    Critical mass where m_a^2/(2ν) ~ (1/2) g_aγ B_⊥
    Returns: m_a,crit [GeV]
    """
    B_GeV2 = microGauss_to_GeV2(B_perp_uG)
    return np.sqrt(nu_GeV * g_agamma * B_GeV2)

def oscillation_length_GeV(m_a_GeV, nu_GeV, Delta_pl_GeV):
    """
    Oscillation length in natural units [GeV^-1]
    For two-level system: L_osc = 2π / |Δm^2/(2ν)|
    
    Effective mass difference includes plasma:
    Δm_eff^2 = |m_a^2 - ω_p^2|
    """
    Delta_a_val = Delta_a(m_a_GeV, nu_GeV)
    # Effective detuning
    Delta_eff = abs(Delta_a_val - Delta_pl_GeV)
    if Delta_eff == 0:
        return np.inf
    return 2 * np.pi / Delta_eff

def GeV_to_kpc(GeV_inv):
    """Convert GeV^-1 to kpc"""
    return GeV_inv / kpc_to_inverseGeV(1.0)

# Typical values
n_e_constant = 7.68e-45  # GeV^3 (constant model)
n_e_typical_cm3 = 0.03   # cm^-3 (typical ISM)
n_e_typical_GeV3 = n_e_typical_cm3 * (5.0677307e13)**-3

B_typical_uG = 2.0  # μG (typical field)
g_agamma_typical = 1e-10  # GeV^-1
nu_10GHz = GHz_to_GeV(10.0)  # 10 GHz
nu_100GHz = GHz_to_GeV(100.0)  # 100 GHz

print("=" * 60)
print("DIMENSIONAL ANALYSIS: Critical Axion Masses")
print("=" * 60)

# Calculate critical masses
print("\n1. PLASMA-MATCHED MASS (m_a ~ ω_p)")
print("-" * 60)
for label, n_e in [("Constant model", n_e_constant), 
                    ("Typical ISM", n_e_typical_GeV3)]:
    m_crit = critical_mass_plasma(n_e)
    omega_p = np.sqrt(omega_plasma_sq(n_e))
    print(f"\n{label}:")
    print(f"  n_e = {n_e:.2e} GeV^3 = {n_e * 5.0677307e13**3:.2e} cm^-3")
    print(f"  ω_p = {omega_p:.2e} GeV = {omega_p * 1e9:.2e} eV")
    print(f"  m_a,crit = {m_crit:.2e} GeV = {m_crit * 1e9:.2e} eV")

print("\n\n2. MIXING-MATCHED MASS (m_a ~ √(2ν g_aγ B_⊥))")
print("-" * 60)
for nu_label, nu in [("10 GHz", nu_10GHz), ("100 GHz", nu_100GHz)]:
    m_crit = critical_mass_mixing(nu, g_agamma_typical, B_typical_uG)
    print(f"\n{nu_label} (ν = {nu:.2e} GeV):")
    print(f"  B_⊥ = {B_typical_uG} μG")
    print(f"  g_aγ = {g_agamma_typical} GeV^-1")
    print(f"  m_a,crit = {m_crit:.2e} GeV = {m_crit * 1e9:.2e} eV")

print("\n\n3. OSCILLATION LENGTHS")
print("-" * 60)
Delta_pl = Delta_plasma(n_e_typical_GeV3, nu_10GHz)

mass_range = np.logspace(-30, -18, 100)  # GeV
L_osc = [GeV_to_kpc(oscillation_length_GeV(m, nu_10GHz, Delta_pl)) 
         for m in mass_range]

print(f"\nFor ν = 10 GHz, n_e = {n_e_typical_GeV3:.2e} GeV^3:")
print(f"  Δ_pl = {Delta_pl:.2e} GeV")

# Find where L_osc ~ 1 kpc, 10 kpc, 100 kpc
for L_target in [1.0, 10.0, 100.0]:
    idx = np.argmin(np.abs(np.array(L_osc) - L_target))
    m_at_L = mass_range[idx]
    print(f"\n  L_osc ≈ {L_target} kpc when m_a ≈ {m_at_L:.2e} GeV = {m_at_L*1e9:.2e} eV")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Oscillation length vs mass
ax = axes[0, 0]
mass_eV = mass_range * 1e9
ax.loglog(mass_eV, L_osc, 'b-', linewidth=2)
ax.axhline(1, color='r', linestyle='--', alpha=0.5, label='1 kpc')
ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='10 kpc')
ax.axhline(100, color='g', linestyle='--', alpha=0.5, label='100 kpc')
ax.set_xlabel(r'Axion Mass $m_a$ (eV)', fontsize=12)
ax.set_ylabel(r'Oscillation Length $L_{osc}$ (kpc)', fontsize=12)
ax.set_title(r'Oscillation Length vs Mass ($\nu=10$ GHz)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Energy scales comparison
ax = axes[0, 1]
mass_eV_plot = np.logspace(-15, -6, 200)
mass_GeV_plot = mass_eV_plot * 1e-9

Delta_a_vals = np.abs([Delta_a(m, nu_10GHz) for m in mass_GeV_plot])
Delta_pl_val = abs(Delta_plasma(n_e_typical_GeV3, nu_10GHz))
Delta_mix = 0.5 * g_agamma_typical * microGauss_to_GeV2(B_typical_uG)

ax.loglog(mass_eV_plot, Delta_a_vals * 1e9, 'b-', linewidth=2, label=r'$|\Delta_a| = m_a^2/(2\nu)$')
ax.axhline(Delta_pl_val * 1e9, color='r', linestyle='--', linewidth=2, label=r'$|\Delta_{pl}|$')
ax.axhline(Delta_mix * 1e9, color='g', linestyle='--', linewidth=2, label=r'$\Delta_{a\gamma}$')
ax.set_xlabel(r'Axion Mass $m_a$ (eV)', fontsize=12)
ax.set_ylabel(r'Energy Scale (eV)', fontsize=12)
ax.set_title('Energy Scales in Hamiltonian', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: Critical masses vs frequency
ax = axes[1, 0]
frequencies_GHz = np.linspace(1, 200, 100)
frequencies_GeV = [GHz_to_GeV(f) for f in frequencies_GHz]

m_crit_mix = [critical_mass_mixing(nu, g_agamma_typical, B_typical_uG) * 1e9 
              for nu in frequencies_GeV]
m_crit_pl = critical_mass_plasma(n_e_typical_GeV3) * 1e9  # Constant

ax.semilogy(frequencies_GHz, m_crit_mix, 'b-', linewidth=2, 
            label=r'Mixing-matched: $m_a \sim \sqrt{2\nu g_{a\gamma}B}$')
ax.axhline(m_crit_pl, color='r', linestyle='--', linewidth=2, 
           label=r'Plasma-matched: $m_a \sim \omega_p$')
ax.set_xlabel(r'Frequency $\nu$ (GHz)', fontsize=12)
ax.set_ylabel(r'Critical Mass $m_{a,crit}$ (eV)', fontsize=12)
ax.set_title('Critical Masses vs Frequency', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 4: Regime diagram
ax = axes[1, 1]
mass_range_regime = np.logspace(-15, -6, 300)
freq_range_regime = np.logspace(0, 3, 300)  # 1 to 1000 GHz

M, F = np.meshgrid(mass_range_regime, freq_range_regime)

# Calculate which term dominates
regime = np.zeros_like(M)
for i, freq_GHz in enumerate(freq_range_regime):
    nu = GHz_to_GeV(freq_GHz)
    for j, mass_eV in enumerate(mass_range_regime):
        m_GeV = mass_eV * 1e-9
        
        Da = abs(Delta_a(m_GeV, nu))
        Dpl = abs(Delta_plasma(n_e_typical_GeV3, nu))
        Dmix = 0.5 * g_agamma_typical * microGauss_to_GeV2(B_typical_uG)
        
        if Da < 0.1 * Dpl and Da < 0.1 * Dmix:
            regime[i, j] = 0  # Mass irrelevant
        elif Da > 10 * Dpl and Da > 10 * Dmix:
            regime[i, j] = 2  # Mass dominated
        else:
            regime[i, j] = 1  # Transition regime

im = ax.contourf(M, F, regime, levels=[-0.5, 0.5, 1.5, 2.5], 
                 colors=['lightblue', 'yellow', 'lightcoral'], alpha=0.7)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Axion Mass $m_a$ (eV)', fontsize=12)
ax.set_ylabel(r'Frequency $\nu$ (GHz)', fontsize=12)
ax.set_title('Regime Diagram', fontsize=12)
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.7, label='Mass Irrelevant'),
    Patch(facecolor='yellow', alpha=0.7, label='Transition'),
    Patch(facecolor='lightcoral', alpha=0.7, label='Mass Dominated')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
# plt.savefig("dimensional_analysis.png", dpi=150)
# print("\n\nPlot saved as 'dimensional_analysis.png'")
plt.show()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nFor typical parameters (ν=10 GHz, B=2μG, g_aγ=10^-10 GeV^-1):")
print(f"  • Plasma-matched mass: m_a ~ {critical_mass_plasma(n_e_typical_GeV3)*1e9:.2e} eV")
print(f"  • Mixing-matched mass: m_a ~ {critical_mass_mixing(nu_10GHz, g_agamma_typical, B_typical_uG)*1e9:.2e} eV")
print(f"\nMass becomes relevant when m_a ≳ 10^-14 eV for typical ISM conditions")
print(f"Mass dominates when m_a ≳ 10^-12 eV")