# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from si_prefix import si_format
import types

# Import Frequency Sweep Data for DCOs - TT condition
tmp = sp.io.loadmat('PSS_freq_sweep_tt_60_1p0V.mat')
raw_vctrl_tt = tmp['vctrl'].squeeze()  # Voltage control vector
raw_freq_dco0_tt = tmp['freq_dco0'].squeeze()  # Frequency offset vector for DCO0
raw_freq_dco1_tt = tmp['freq_dco1'].squeeze()  # Frequency offset vector for DCO1 
raw_freq_dco2_tt = tmp['freq_dco2'].squeeze()  # Frequency offset vector for DCO2 
raw_freq_dco3_tt = tmp['freq_dco3'].squeeze()  # Frequency offset vector for DCO3 

# Import Frequency Sweep Data for DCOs - FF condition
tmp = sp.io.loadmat('PSS_freq_sweep_ff_m40_1p1V.mat')
raw_vctrl_ff = tmp['vctrl'].squeeze()  # Voltage control vector
raw_freq_dco0_ff = tmp['freq_dco0'].squeeze()  # Frequency offset vector for DCO0
raw_freq_dco1_ff = tmp['freq_dco1'].squeeze()  # Frequency offset vector for DCO1 
raw_freq_dco2_ff = tmp['freq_dco2'].squeeze()  # Frequency offset vector for DCO2 
raw_freq_dco3_ff = tmp['freq_dco3'].squeeze()  # Frequency offset vector for DCO3 

# Import Frequency Sweep Data for DCOs - SS condition
tmp = sp.io.loadmat('PSS_freq_sweep_ss_125_0p9V.mat')
raw_vctrl_ss = tmp['vctrl'].squeeze()  # Voltage control vector
raw_freq_dco0_ss = tmp['freq_dco0'].squeeze()  # Frequency offset vector for DCO0
raw_freq_dco1_ss = tmp['freq_dco1'].squeeze()  # Frequency offset vector for DCO1 
raw_freq_dco2_ss = tmp['freq_dco2'].squeeze()  # Frequency offset vector for DCO2 
raw_freq_dco3_ss = tmp['freq_dco3'].squeeze()  # Frequency offset vector for DCO3 

# Interpolate to common frequency vector
samples = 100
vctrl = np.linspace(np.max([np.min(raw_vctrl_tt), np.min(raw_vctrl_ff), np.min(raw_vctrl_ss)]), 
                    np.min([np.max(raw_vctrl_tt), np.max(raw_vctrl_ff), np.max(raw_vctrl_ss)]), samples)

# TT condition
freq_dco0_tt = np.interp(vctrl, raw_vctrl_tt, raw_freq_dco0_tt)
freq_dco1_tt = np.interp(vctrl, raw_vctrl_tt, raw_freq_dco1_tt)
freq_dco2_tt = np.interp(vctrl, raw_vctrl_tt, raw_freq_dco2_tt)
freq_dco3_tt = np.interp(vctrl, raw_vctrl_tt, raw_freq_dco3_tt)

# FF condition
freq_dco0_ff = np.interp(vctrl, raw_vctrl_ff, raw_freq_dco0_ff)
freq_dco1_ff = np.interp(vctrl, raw_vctrl_ff, raw_freq_dco1_ff)
freq_dco2_ff = np.interp(vctrl, raw_vctrl_ff, raw_freq_dco2_ff)
freq_dco3_ff = np.interp(vctrl, raw_vctrl_ff, raw_freq_dco3_ff)

# SS condition
freq_dco0_ss = np.interp(vctrl, raw_vctrl_ss, raw_freq_dco0_ss)
freq_dco1_ss = np.interp(vctrl, raw_vctrl_ss, raw_freq_dco1_ss)
freq_dco2_ss = np.interp(vctrl, raw_vctrl_ss, raw_freq_dco2_ss)
freq_dco3_ss = np.interp(vctrl, raw_vctrl_ss, raw_freq_dco3_ss)



# Import Phase Noise Sweep Data for DCOs - TT condition
tmp = sp.io.loadmat('PSS_pnoise1M_sweep_tt_60_1p0V.mat')
raw_vctrl_pn_tt = tmp['vctrl'].squeeze()  # Voltage control vector
raw_pnoise_dco0_tt = tmp['pn_dco0'].squeeze()  # Phase noise for DCO0
raw_pnoise_dco1_tt = tmp['pn_dco1'].squeeze()  # Phase noise for DCO1 
raw_pnoise_dco2_tt = tmp['pn_dco2'].squeeze()  # Phase noise for DCO2 
raw_pnoise_dco3_tt = tmp['pn_dco3'].squeeze()  # Phase noise for DCO3 

# Import Phase Noise Sweep Data for DCOs - FF condition
tmp = sp.io.loadmat('PSS_pnoise1M_sweep_ff_m40_1p1V.mat')
raw_vctrl_pn_ff = tmp['vctrl'].squeeze()  # Voltage control vector
raw_pnoise_dco0_ff = tmp['pn_dco0'].squeeze()  # Phase noise for DCO0
raw_pnoise_dco1_ff = tmp['pn_dco1'].squeeze()  # Phase noise for DCO1 
raw_pnoise_dco2_ff = tmp['pn_dco2'].squeeze()  # Phase noise for DCO2 
raw_pnoise_dco3_ff = tmp['pn_dco3'].squeeze()  # Phase noise for DCO3 

# Import Phase Noise Sweep Data for DCOs - SS condition
tmp = sp.io.loadmat('PSS_pnoise1M_sweep_ss_125_0p9V.mat')
raw_vctrl_pn_ss = tmp['vctrl'].squeeze()  # Voltage control vector
raw_pnoise_dco0_ss = tmp['pn_dco0'].squeeze()  # Phase noise for DCO0
raw_pnoise_dco1_ss = tmp['pn_dco1'].squeeze()  # Phase noise for DCO1 
raw_pnoise_dco2_ss = tmp['pn_dco2'].squeeze()  # Phase noise for DCO2 
raw_pnoise_dco3_ss = tmp['pn_dco3'].squeeze()  # Phase noise for DCO3 

# Interpolate phase noise to common voltage vector
pnoise_dco0_tt = np.interp(vctrl, raw_vctrl_pn_tt, raw_pnoise_dco0_tt)
pnoise_dco1_tt = np.interp(vctrl, raw_vctrl_pn_tt, raw_pnoise_dco1_tt)
pnoise_dco2_tt = np.interp(vctrl, raw_vctrl_pn_tt, raw_pnoise_dco2_tt)
pnoise_dco3_tt = np.interp(vctrl, raw_vctrl_pn_tt, raw_pnoise_dco3_tt)

pnoise_dco0_ff = np.interp(vctrl, raw_vctrl_pn_ff, raw_pnoise_dco0_ff)
pnoise_dco1_ff = np.interp(vctrl, raw_vctrl_pn_ff, raw_pnoise_dco1_ff)
pnoise_dco2_ff = np.interp(vctrl, raw_vctrl_pn_ff, raw_pnoise_dco2_ff)
pnoise_dco3_ff = np.interp(vctrl, raw_vctrl_pn_ff, raw_pnoise_dco3_ff)

pnoise_dco0_ss = np.interp(vctrl, raw_vctrl_pn_ss, raw_pnoise_dco0_ss)
pnoise_dco1_ss = np.interp(vctrl, raw_vctrl_pn_ss, raw_pnoise_dco1_ss)
pnoise_dco2_ss = np.interp(vctrl, raw_vctrl_pn_ss, raw_pnoise_dco2_ss)
pnoise_dco3_ss = np.interp(vctrl, raw_vctrl_pn_ss, raw_pnoise_dco3_ss)


# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'

# Plot Frequency vs Control Voltage
plt.figure(figsize=(6, 4), dpi=200)
plt.minorticks_on()
plt.axhline(y=2.5, color='black', linestyle='--', linewidth=3)
#plt.text(np.max(vctrl*1e3) - 50, 2.5, '2.5GHz', ha='right', va='center', color='black', fontweight='bold')
plt.plot(vctrl*1e3, freq_dco0_tt / 1e9, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, freq_dco1_tt / 1e9, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, freq_dco2_tt / 1e9, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, freq_dco3_tt / 1e9, label='DCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Frequency (GHz)', fontweight='bold')
#plt.title('DCO Frequency vs Control Voltage')   
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
#plt.ylim([0.25, 3.5])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('vctrl_freq.pdf',dpi=400)  
plt.show()

# Plot Phase Noise vs Control Voltage
plt.figure(figsize=(6, 4), dpi=200)
plt.minorticks_on()
plt.plot(vctrl*1e3, pnoise_dco0_tt, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco1_tt, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco2_tt, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco3_tt, label='DCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Phase Noise (dBc/Hz)', fontweight='bold')
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('vctrl_pnoise.pdf',dpi=400)  
plt.show()  



# Combined plot: Frequency and Phase Noise vs Control Voltage side by side
plt.figure(figsize=(9, 4), dpi=200)

# First subplot: Frequency vs Control Voltage
plt.subplot(1, 2, 1)
plt.minorticks_on()
plt.axhline(y=2.5, color='black', linestyle='--', linewidth=3)
plt.plot(vctrl*1e3, freq_dco0_tt / 1e9, label='DVCO0', linewidth=3)
plt.plot(vctrl*1e3, freq_dco1_tt / 1e9, label='DVCO1', linewidth=3)
plt.plot(vctrl*1e3, freq_dco2_tt / 1e9, label='DVCO2', linewidth=3)
plt.plot(vctrl*1e3, freq_dco3_tt / 1e9, label='DVCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Frequency (GHz)', fontweight='bold')
#plt.title('DCO Frequency vs Control Voltage')
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Second subplot: Phase Noise vs Control Voltage
plt.subplot(1, 2, 2)
plt.minorticks_on()
plt.plot(vctrl*1e3, pnoise_dco0_tt, label='DVCO0', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco1_tt, label='DVCO1', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco2_tt, label='DVCO2', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco3_tt, label='DVCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Phase Noise (dBc/Hz)', fontweight='bold')
#plt.title('Phase Noise vs Control Voltage')
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.savefig('vco_freq_pn_tt.pdf', dpi=400)
plt.show()

# Contour Plot: Frequency vs Control Voltage and DCO - TT, FF, SS conditions
# Create 2D arrays for contour plot
dco_indices = np.array([0, 1, 2, 3])

# TT condition
freq_matrix_tt = np.array([freq_dco0_tt / 1e9, freq_dco1_tt / 1e9, freq_dco2_tt / 1e9, freq_dco3_tt / 1e9])

# FF condition
freq_matrix_ff = np.array([freq_dco0_ff / 1e9, freq_dco1_ff / 1e9, freq_dco2_ff / 1e9, freq_dco3_ff / 1e9])

# SS condition
freq_matrix_ss = np.array([freq_dco0_ss / 1e9, freq_dco1_ss / 1e9, freq_dco2_ss / 1e9, freq_dco3_ss / 1e9])

# Create meshgrid
X, Y = np.meshgrid(vctrl * 1e3, dco_indices)

# Create subplot figure with all three conditions
fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=200, sharey=True)

# SS condition subplot
contour_ss = axes[0].contourf(X, Y, freq_matrix_ss, levels=10, cmap='Reds', alpha=0.7)
contour_lines_ss = axes[0].contour(X, Y, freq_matrix_ss, levels=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], colors='darkred', linewidths=2)
axes[0].clabel(contour_lines_ss, inline=True, fontsize=10, fmt='%.2f GHz')
axes[0].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[0].set_ylabel('DCO Index', fontweight='bold')
axes[0].set_yticks(dco_indices)
axes[0].set_yticklabels([f'DCO{i}' for i in dco_indices])
axes[0].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for SS condition
from matplotlib.lines import Line2D
legend_ss = [Line2D([0], [0], color='darkred', lw=2, label='Slow-Slow (SS) Frequency')]
axes[0].legend(handles=legend_ss, loc='upper right', fontsize=10)

# TT condition subplot
contour_tt = axes[1].contourf(X, Y, freq_matrix_tt, levels=10, cmap='Greens', alpha=0.7)
contour_lines_tt = axes[1].contour(X, Y, freq_matrix_tt, levels=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], colors='darkgreen', linewidths=2)
axes[1].clabel(contour_lines_tt, inline=True, fontsize=10, fmt='%.2f GHz')
axes[1].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[1].set_yticks(dco_indices)
axes[1].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for TT condition
legend_tt = [Line2D([0], [0], color='darkgreen', lw=2, label='Typical-Typical (TT) Frequency')]
axes[1].legend(handles=legend_tt, loc='upper right', fontsize=10)

# FF condition subplot
contour_ff = axes[2].contourf(X, Y, freq_matrix_ff, levels=10, cmap='Blues', alpha=0.7)
contour_lines_ff = axes[2].contour(X, Y, freq_matrix_ff, levels=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], colors='darkblue', linewidths=2)
axes[2].clabel(contour_lines_ff, inline=True, fontsize=10, fmt='%.2f GHz')
axes[2].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[2].set_yticks(dco_indices)
axes[2].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for FF condition
legend_ff = [Line2D([0], [0], color='darkblue', lw=2, label='Fast-Fast (FF) Frequency')]
axes[2].legend(handles=legend_ff, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('vco_contour_freq.pdf', dpi=400)
plt.show()

# Contour Plot: Phase Noise vs Control Voltage and DCO - TT, FF, SS conditions
# Create 2D arrays for phase noise contour plot
pnoise_matrix_tt = np.array([pnoise_dco0_tt, pnoise_dco1_tt, pnoise_dco2_tt, pnoise_dco3_tt])
pnoise_matrix_ff = np.array([pnoise_dco0_ff, pnoise_dco1_ff, pnoise_dco2_ff, pnoise_dco3_ff])
pnoise_matrix_ss = np.array([pnoise_dco0_ss, pnoise_dco1_ss, pnoise_dco2_ss, pnoise_dco3_ss])

# Create subplot figure with all three conditions for phase noise
fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=200, sharey=True)

# SS condition subplot
contour_ss = axes[0].contourf(X, Y, pnoise_matrix_ss, levels=10, cmap='Reds', alpha=0.7)
contour_lines_ss = axes[0].contour(X, Y, pnoise_matrix_ss, colors='darkred', linewidths=2)
axes[0].clabel(contour_lines_ss, inline=True, fontsize=10, fmt='%.0f dBc/Hz')
axes[0].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[0].set_ylabel('DCO Index', fontweight='bold')
axes[0].set_yticks(dco_indices)
axes[0].set_yticklabels([f'DCO{i}' for i in dco_indices])
axes[0].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for SS condition
legend_ss_pn = [Line2D([0], [0], color='darkred', lw=2, label='Slow-Slow (SS) Phase Noise')]
axes[0].legend(handles=legend_ss_pn, loc='upper right', fontsize=10)

# TT condition subplot
contour_tt = axes[1].contourf(X, Y, pnoise_matrix_tt, levels=10, cmap='Greens', alpha=0.7)
contour_lines_tt = axes[1].contour(X, Y, pnoise_matrix_tt, colors='darkgreen', linewidths=2)
axes[1].clabel(contour_lines_tt, inline=True, fontsize=10, fmt='%.0f dBc/Hz')
axes[1].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[1].set_yticks(dco_indices)
axes[1].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for TT condition
legend_tt_pn = [Line2D([0], [0], color='darkgreen', lw=2, label='Typical-Typical (TT) Phase Noise')]
axes[1].legend(handles=legend_tt_pn, loc='upper right', fontsize=10)

# FF condition subplot
contour_ff = axes[2].contourf(X, Y, pnoise_matrix_ff, levels=10, cmap='Blues', alpha=0.7)
contour_lines_ff = axes[2].contour(X, Y, pnoise_matrix_ff, colors='darkblue', linewidths=2)
axes[2].clabel(contour_lines_ff, inline=True, fontsize=10, fmt='%.0f dBc/Hz')
axes[2].set_xlabel('Control Voltage (mV)', fontweight='bold')
axes[2].set_yticks(dco_indices)
axes[2].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.3)
# Add legend for FF condition
legend_ff_pn = [Line2D([0], [0], color='darkblue', lw=2, label='Fast-Fast (FF) Phase Noise')]
axes[2].legend(handles=legend_ff_pn, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('vco_contour_pnoise.pdf', dpi=400)
plt.show()

