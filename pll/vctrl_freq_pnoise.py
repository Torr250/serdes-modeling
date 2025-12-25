# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from si_prefix import si_format
import types

# Import Frequency Sweep Data for DCOs
tmp = sp.io.loadmat('PSS_freq_sweep_tt_60_1p0V.mat')
raw_vctrl = tmp['vctrl'].squeeze()  # Voltage control vector
raw_freq_dco0 = tmp['freq_dco0'].squeeze()  # Frequency offset vector for DCO0
raw_freq_dco1 = tmp['freq_dco1'].squeeze()  # Frequency offset vector for DCO1 
raw_freq_dco2 = tmp['freq_dco2'].squeeze()  # Frequency offset vector for DCO2 
raw_freq_dco3 = tmp['freq_dco3'].squeeze()  # Frequency offset vector for DCO3 

# Interpolate to PLL frequency vector
samples = 100
vctrl = np.linspace(np.min(raw_vctrl), np.max(raw_vctrl), samples)
freq_dco0 = np.interp(vctrl, raw_vctrl, raw_freq_dco0)
freq_dco1 = np.interp(vctrl, raw_vctrl, raw_freq_dco1)
freq_dco2 = np.interp(vctrl, raw_vctrl, raw_freq_dco2)
freq_dco3 = np.interp(vctrl, raw_vctrl, raw_freq_dco3)

# Import Phase Noise Sweep Data for DCOs
tmp = sp.io.loadmat('PSS_pnoise_sweep_tt_60_1p0V.mat')
raw_vctrl = tmp['vctrl'].squeeze()  # Voltage control vector
raw_pnoise_dco0 = tmp['pn_dco0'].squeeze()  # Frequency offset vector for DCO0
raw_pnoise_dco1 = tmp['pn_dco1'].squeeze()  # Frequency offset vector for DCO1 
raw_pnoise_dco2 = tmp['pn_dco2'].squeeze()  # Frequency offset vector for DCO2 
raw_pnoise_dco3 = tmp['pn_dco3'].squeeze()  # Frequency offset vector for DCO3 
# Interpolate to PLL frequency vector
pnoise_dco0 = np.interp(vctrl, raw_vctrl, raw_pnoise_dco0)
pnoise_dco1 = np.interp(vctrl, raw_vctrl, raw_pnoise_dco1)
pnoise_dco2 = np.interp(vctrl, raw_vctrl, raw_pnoise_dco2)
pnoise_dco3 = np.interp(vctrl, raw_vctrl, raw_pnoise_dco3)


# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'

# Plot Frequency vs Control Voltage
plt.figure(figsize=(6, 4), dpi=200)
plt.minorticks_on()
plt.axhline(y=2.5, color='black', linestyle='--', linewidth=3)
#plt.text(np.max(vctrl*1e3) - 50, 2.5, '2.5GHz', ha='right', va='center', color='black', fontweight='bold')
plt.plot(vctrl*1e3, freq_dco0 / 1e9, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, freq_dco1 / 1e9, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, freq_dco2 / 1e9, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, freq_dco3 / 1e9, label='DCO3', linewidth=3)
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
plt.plot(vctrl*1e3, pnoise_dco0, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco1, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco2, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco3, label='DCO3', linewidth=3)
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
plt.plot(vctrl*1e3, freq_dco0 / 1e9, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, freq_dco1 / 1e9, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, freq_dco2 / 1e9, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, freq_dco3 / 1e9, label='DCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Frequency (GHz)', fontweight='bold')
#plt.title('DCO Frequency vs Control Voltage')
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Second subplot: Phase Noise vs Control Voltage
plt.subplot(1, 2, 2)
plt.minorticks_on()
plt.plot(vctrl*1e3, pnoise_dco0, label='DCO0', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco1, label='DCO1', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco2, label='DCO2', linewidth=3)
plt.plot(vctrl*1e3, pnoise_dco3, label='DCO3', linewidth=3)
plt.xlabel('Control Voltage (mV)', fontweight='bold')
plt.ylabel('Phase Noise (dBc/Hz)', fontweight='bold')
#plt.title('Phase Noise vs Control Voltage')
plt.xlim([np.min(vctrl*1e3), np.max(vctrl*1e3)])
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('combined_plot.pdf', dpi=400)
plt.show()

