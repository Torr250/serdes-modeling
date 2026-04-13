# Import CSV and plot VCtrl of PLL
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

# Read the CSV file with specified column names
csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/tran_vctrl_ss_ff_tt_pex.csv'
df = pd.read_csv(csv_path, header=None, names=['time_ff', 'voltage_ff', 'time_tt', 'voltage_tt', 'time_ss', 'voltage_ss'])

# Replace whitespace-only strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Generate arrays for each column, eliminating NaN (non-existing data)
time_ff = df['time_ff'].dropna().values
voltage_ff = df['voltage_ff'].dropna().values
time_tt = df['time_tt'].dropna().values
voltage_tt = df['voltage_tt'].dropna().values
time_ss = df['time_ss'].dropna().values
voltage_ss = df['voltage_ss'].dropna().values
# Convert string arrays to float arrays
t_ff = time_ff.astype(float)
v_ff = voltage_ff.astype(float)
t_tt = time_tt.astype(float)
v_tt = voltage_tt.astype(float)
t_ss = time_ss.astype(float)
v_ss = voltage_ss.astype(float)
# Print lengths of each array to verify
print(f"time_ff length: {len(t_ff)}")
print(f"v_ff length: {len(v_ff)}")
print(f"time_tt length: {len(t_tt)}")
print(f"v_tt length: {len(v_tt)}")
print(f"time_ss length: {len(t_ss)}")
print(f"v_ss length: {len(v_ss)}")


# Plot VCtrl for the three process corners
# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.figure(figsize=(7, 5))
plt.plot(t_ff*1e6, v_ff*1e3, label='FF 1.1V -40°C', linewidth=1, color='blue')
plt.plot(t_tt*1e6, v_tt*1e3, label='TT 1.0V 60°C', linewidth=1, color='green')
plt.plot(t_ss*1e6, v_ss*1e3, label='SS 0.9V 125°C', linewidth=1, color='red')
plt.xlabel('Time (us)', fontweight='bold')
plt.ylabel('VCtrl (mV)', fontweight='bold')
plt.xlim(0, 1)  # Limit x-axis to first 2 us
plt.ylim(1000,150)  # Invert y-axis for better visibility
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('vctrl_ff_ss_tt.pdf', dpi=400)
plt.show()
