# Import CSV data and then generate plots of the Phase noise 
# Read the CSV file with specified column names
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/tran_pnoise_ss_ff_tt_pex.csv'
df = pd.read_csv(csv_path, header=None, names=['frequency_ff', 'pnoise_ff', 'frequency_tt', 'pnoise_tt', 'frequency_ss', 'pnoise_ss'])

# Replace whitespace-only strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Generate arrays for each column, eliminating NaN (non-existing data)
frequency_ff = df['frequency_ff'].dropna().values
pnoise_ff = df['pnoise_ff'].dropna().values
frequency_tt = df['frequency_tt'].dropna().values
pnoise_tt = df['pnoise_tt'].dropna().values
frequency_ss = df['frequency_ss'].dropna().values
pnoise_ss = df['pnoise_ss'].dropna().values
# Convert string arrays to float arrays
f_ff = frequency_ff.astype(float)
pn_ff = pnoise_ff.astype(float)
f_tt = frequency_tt.astype(float)
pn_tt = pnoise_tt.astype(float)
f_ss = frequency_ss.astype(float)
pn_ss = pnoise_ss.astype(float)
# Print lengths of each array to verify
print(f"frequency_ff length: {len(f_ff)}")
print(f"pnoise_ff length: {len(pn_ff)}")
print(f"frequency_tt length: {len(f_tt)}")
print(f"pnoise_tt length: {len(pn_tt)}")
print(f"frequency_ss length: {len(f_ss)}")
print(f"pnoise_ss length: {len(pn_ss)}")

# Plot Phase Noise for the three process corners
# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
# Plot the phase noise
plt.figure(figsize=(8, 6))
plt.minorticks_on()
plt.plot(f_ff, pn_ff, label='FF Phase Noise', linewidth=3, color='blue')
plt.plot(f_ss, pn_ss, label='SS Phase Noise', linewidth=3, color='red')
plt.plot(f_tt, pn_tt, label='TT Phase Noise', linewidth=3, color='green')
plt.xlabel('Frequency (Hz)', fontweight='bold')
plt.ylabel('Phase Noise (dBc/Hz)', fontweight='bold')
plt.xlim(min(f_ff[1], f_ss[1], f_tt[1]), max(f_ff[-1], f_ss[-1], f_tt[-1]))  # Limit x-axis from 1 kHz to 1 GHz
plt.ylim(-180, -80)  # Limit y-axis for better visibility
plt.xscale('log')
plt.yscale('linear')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('pnoise_ff_ss_tt.pdf', dpi=400)
plt.show()
