# import CSV file and plot histogram of jitter values for three process corners: FF, TT, SS
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
# Read the CSV file with specified column names
csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/tran_absjitterF_ss_ff_tt_pex.csv'
df = pd.read_csv(csv_path, header=None, names=['time_ff', 'jitter_ff', 'time_tt', 'jitter_tt', 'time_ss', 'jitter_ss'])

# Generate arrays for each column, eliminating NaN (non-existing data)
jitter_ff = df['jitter_ff'].dropna().values
jitter_tt = df['jitter_tt'].dropna().values
jitter_ss = df['jitter_ss'].dropna().values

# Convert string arrays to float arrays
j_ff = jitter_ff.astype(float)
j_tt = jitter_tt.astype(float)
j_ss = jitter_ss.astype(float)

# Calculate and print mean and standard deviation of jitter for each process corner
mean_j_ff = np.mean(j_ff)
std_j_ff = np.std(j_ff)
mean_j_tt = np.mean(j_tt)
std_j_tt = np.std(j_tt)
mean_j_ss = np.mean(j_ss)
std_j_ss = np.std(j_ss)
print(f"FF Jitter: Mean = {mean_j_ff*1e12:.2f} ps, Std Dev = {std_j_ff*1e12:.2f} ps")
print(f"TT Jitter: Mean = {mean_j_tt*1e12:.2f} ps, Std Dev = {std_j_tt*1e12:.2f} ps")
print(f"SS Jitter: Mean = {mean_j_ss*1e12:.2f} ps, Std Dev = {std_j_ss*1e12:.2f} ps")

# Calculate peak-to-peak jitter for each process corner
ptp_j_ff = np.ptp(j_ff)
ptp_j_tt = np.ptp(j_tt)
ptp_j_ss = np.ptp(j_ss)
print(f"FF Jitter: Peak-to-Peak = {ptp_j_ff*1e12:.2f} ps")
print(f"TT Jitter: Peak-to-Peak = {ptp_j_tt*1e12:.2f} ps")
print(f"SS Jitter: Peak-to-Peak = {ptp_j_ss*1e12:.2f} ps")


# Plot histogram of jitter values for the three process corners
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.figure(figsize=(7, 5))
plt.minorticks_on()
plt.hist(j_ss*1e12, bins=100, alpha=0.7, label=r'SS 0.9V 125°C: $\sigma$: {:.2f} ps'.format(std_j_ss*1e12), color='red', histtype="step", linewidth=3)
plt.hist(j_ff*1e12, bins=100, alpha=0.7, label=r'FF 1.1V -40°C: $\sigma$: {:.2f} ps'.format(std_j_ff*1e12), color='blue', histtype="step", linewidth=3)
plt.hist(j_tt*1e12, bins=100, alpha=0.7, label=r'TT 1.0V  60°C: $\sigma$: {:.2f} ps'.format(std_j_tt*1e12), color='green', histtype="step", linewidth=3)
plt.xlabel('Jitter (ps)', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('absjitter_ff_ss_tt.pdf', dpi=400)
plt.show()