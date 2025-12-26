#Spectrum plots from csv data
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

# Load spectrum data from CSV files
spectrum_ff = pd.read_csv('spectrum_ff.csv')
spectrum_tt = pd.read_csv('spectrum_tt.csv')
spectrum_ss = pd.read_csv('spectrum_ss.csv')
frequencies = spectrum_ff['Frequency (Hz)'].values
magnitude_ff = spectrum_ff['Magnitude (dB)'].values
magnitude_tt = spectrum_tt['Magnitude (dB)'].values
magnitude_ss = spectrum_ss['Magnitude (dB)'].values
fosc = 2.5e9  # Oscillator frequency
fref = 150e6  # Reference frequency

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
# Plot the spectra
plt.figure(figsize=(10, 6))
plt.minorticks_on()
plt.plot(frequencies*1e-9, magnitude_ff, label='FF Spectrum', linewidth=3, color='blue')
plt.plot(frequencies*1e-9, magnitude_ss, label='SS Spectrum', linewidth=3, color='red')
plt.plot(frequencies*1e-9, magnitude_tt, label='TT Spectrum', linewidth=3, color='green')
plt.xlabel('Frequency (Hz)', fontweight='bold')
plt.ylabel('Magnitude', fontweight='bold')
plt.xlim((fosc - fref)*1e-9, (fosc + fref)*1e-9)  # Limit x-axis to fosc ± fref
plt.ylim(-100, 10)  # Limit y-axis for better visibility
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('spectrum_ff_ss_tt.pdf', dpi=400)
plt.show()

