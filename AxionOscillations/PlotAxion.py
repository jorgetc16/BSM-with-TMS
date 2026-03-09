import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2
from matplotlib import rc
import matplotlib.patheffects as path_effects

# print an array between 2 and 2.5 with step 0.01


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

csv_file_path_axion_ne2001 = '/home/jortecal/GitHub/BSM-with-TMS/AxionOscillations/data/TMS_axion_sensitivity_ne2001.csv'
Cast_path = '/home/jortecal/GitHub/BSM-with-TMS/AxionOscillations/CAST.txt'
Chandra_H182_path = '/home/jortecal/GitHub/BSM-with-TMS/AxionOscillations/ChandraH182+1643.txt'
Chandra_M87_path = '/home/jortecal/GitHub/BSM-with-TMS/AxionOscillations/Chandra_M87.txt'





def plot_sensitivity(m_a_eV_array, g_lim_array_TMS, g_lim_array_Cast, g_lim_array_Chandra_H182, g_lim_array_Chandra_M87):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    plt.fill_between(m_a_eV_array, g_lim_array_TMS, 1, alpha=0.5, color=colors[0])


    plt.plot(m_a_eV_array, g_lim_array_Chandra_H182, label='Chandra H182+1643', color=colors[2], linewidth=2)
    plt.fill_between(m_a_eV_array, g_lim_array_Chandra_H182, 1, alpha=0.7, color=colors[2])
    plt.text(5e-17, 6e-13, 'Chandra H182+1643', fontsize=18, color=colors[2], ha='center', va='center', path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    plt.plot(m_a_eV_array, g_lim_array_Chandra_M87, label='Chandra M87', color=colors[3], linewidth=2)
    plt.fill_between(m_a_eV_array, g_lim_array_Chandra_M87, 1, alpha=0.7, color=colors[3])
    plt.text(1.5e-17, 2e-12, 'Chandra M87', fontsize=18, color=colors[3], ha='center', va='center', path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    plt.plot(m_a_eV_array, g_lim_array_Cast, label='CAST', color=colors[8], linewidth=2)
    plt.fill_between(m_a_eV_array, g_lim_array_Cast, 1,  alpha=0.7, color=colors[8])
    plt.text(5e-18, 7e-11, 'CAST', fontsize=18, color=colors[8], ha='center', va='center', path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    plt.plot(m_a_eV_array, g_lim_array_TMS, label='TMS Sensitivity', color=colors[0], linewidth=2)
    plt.text(2.3e-17, 3e-13, 'TMS Sensitivity', fontsize=18, color=colors[0], ha='center', va='center', path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Axion Mass $m_a$ (eV)', fontsize=24)
    plt.ylabel(r'Coupling Limit $g_{a\gamma}$ (GeV$^{-1}$)', fontsize=24)
    plt.ylim(1e-13, 1e-10)
    plt.xlim(1e-18, 1e-10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.set_xticks(np.logspace(-18, -10, 5))

    # plt.grid(True, which='both', ls='--', lw=0.5)
    # plt.legend()
    plt.tight_layout()
    # Save results to pdf
    output_file = '/home/jortecal/GitHub/BSM-with-TMS/AxionOscillations/Figures/TMS_axion_sensitivity.pdf'
    plt.savefig(output_file, bbox_inches='tight')

def main():
    # Load TMS sensitivity data
    tms_data = pd.read_csv(csv_file_path_axion_ne2001)
    m_a_eV_file = tms_data['m_a[eV]'].values
    g_lim_file = tms_data['g_agamma[GeV^-1]'].values
    
    ma_array = np.logspace(-18, -10, 200)  # eV
    g_lim_array_TMS = np.interp(ma_array, m_a_eV_file, g_lim_file)

    Chandra_H182_data = np.loadtxt(Chandra_H182_path, comments='#')
    m_a_eV_Chandra_H182_file = Chandra_H182_data[:, 0]
    g_lim_Chandra_H182_file = Chandra_H182_data[:, 1]
    g_lim_array_Chandra_H182 = np.interp(ma_array, m_a_eV_Chandra_H182_file, g_lim_Chandra_H182_file)

    Chandra_M87_data = np.loadtxt(Chandra_M87_path, comments='#')
    m_a_eV_Chandra_M87_file = Chandra_M87_data[:, 0]
    g_lim_Chandra_M87_file = Chandra_M87_data[:, 1]
    g_lim_array_Chandra_M87 = np.interp(ma_array, m_a_eV_Chandra_M87_file, g_lim_Chandra_M87_file)

    Cast_data = np.loadtxt(Cast_path, comments='#')
    m_a_eV_Cast_file = Cast_data[:, 0]
    g_lim_Cast_file = Cast_data[:, 1] 
    g_lim_array_Cast = np.interp(ma_array, m_a_eV_Cast_file, g_lim_Cast_file)



    plot_sensitivity(ma_array, g_lim_array_TMS, g_lim_array_Cast, g_lim_array_Chandra_H182, g_lim_array_Chandra_M87)

try:
    main()
except Exception as e:
    print(f"An error occurred: {e}")
    
    