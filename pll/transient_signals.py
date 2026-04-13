# import CSV file and plot transient signals 
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

# Read the CSV file with specified column names
csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/tran_signals_tt_60_1p0V.csv'
df = pd.read_csv(csv_path, header=None, names=['VCTRL_X', 'VCTRL_Y', 'N1P_X', 'N1P_Y', 'N1N_X', 'N1N_Y', 'VCS_X', 'VCS_Y', 'CLK0_X', 'CLK0_Y', 'CLK180_X', 'CLK180_Y', 'CLK90_X', 'CLK90_Y', 'CLK270_X', 'CLK270_Y', 'DNN_X', 'DNN_Y', 'DNP_X', 'DNP_Y', 'UPP_X', 'UPP_Y', 'UPN_X', 'UPN_Y'], dtype=str)

# Replace whitespace-only strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Generate arrays for each column, eliminating NaN (non-existing data)
VCTRL_X = df['VCTRL_X'].dropna().values
VCTRL_Y = df['VCTRL_Y'].dropna().values
N1P_X = df['N1P_X'].dropna().values
N1P_Y = df['N1P_Y'].dropna().values
N1N_X = df['N1N_X'].dropna().values
N1N_Y = df['N1N_Y'].dropna().values
VCS_X = df['VCS_X'].dropna().values
VCS_Y = df['VCS_Y'].dropna().values
CLK0_X = df['CLK0_X'].dropna().values
CLK0_Y = df['CLK0_Y'].dropna().values
CLK180_X = df['CLK180_X'].dropna().values
CLK180_Y = df['CLK180_Y'].dropna().values
CLK90_X = df['CLK90_X'].dropna().values
CLK90_Y = df['CLK90_Y'].dropna().values
CLK270_X = df['CLK270_X'].dropna().values
CLK270_Y = df['CLK270_Y'].dropna().values
DNN_X = df['DNN_X'].dropna().values
DNN_Y = df['DNN_Y'].dropna().values
DNP_X = df['DNP_X'].dropna().values
DNP_Y = df['DNP_Y'].dropna().values
UPP_X = df['UPP_X'].dropna().values
UPP_Y = df['UPP_Y'].dropna().values
UPN_X = df['UPN_X'].dropna().values
UPN_Y = df['UPN_Y'].dropna().values

# Convert string arrays to float arrays
t_vctrl = VCTRL_X.astype(float)
v_vctrl = VCTRL_Y.astype(float)
t_n1p = N1P_X.astype(float)
v_n1p = N1P_Y.astype(float)
t_n1n = N1N_X.astype(float)
v_n1n = N1N_Y.astype(float)
t_vcs = VCS_X.astype(float)
v_vcs = VCS_Y.astype(float)
t_clk0 = CLK0_X.astype(float)
v_clk0 = CLK0_Y.astype(float)
t_clk180 = CLK180_X.astype(float)
v_clk180 = CLK180_Y.astype(float)
t_clk90 = CLK90_X.astype(float)
v_clk90 = CLK90_Y.astype(float)
t_clk270 = CLK270_X.astype(float)
v_clk270 = CLK270_Y.astype(float)
t_dnn = DNN_X.astype(float)
v_dnn = DNN_Y.astype(float)
t_dnp = DNP_X.astype(float)
v_dnp = DNP_Y.astype(float)
t_upp = UPP_X.astype(float)
v_upp = UPP_Y.astype(float)
t_upn = UPN_X.astype(float)
v_upn = UPN_Y.astype(float)


# Plot CLK0 and CLK180 in different subplots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(7, 7))
ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()

ax1.plot(t_vctrl*1e6, v_vctrl*1e3, label='Vctrl', linewidth=3, color='purple')
ax1.set_ylabel('mV', fontweight='bold',fontsize=11)
ax1.legend(loc='right', fontsize=11)
ax1.set_ylim(641, 644)  # Focus on Vctrl range
ax1.grid(which='both', linestyle='--', linewidth=0.5)

ax2.plot(t_vcs*1e6, v_vcs*1e3, label='VCS', linewidth=3, color='green')
ax2.plot(t_n1p*1e6, v_n1p*1e3, label='N1', linewidth=3, color='red')
ax2.plot(t_n1n*1e6, v_n1n*1e3, label='N2', linewidth=3, color='blue')
ax2.set_ylabel('mV', fontweight='bold',fontsize=11)
ax2.legend(loc='right', fontsize=11)
ax2.grid(which='both', linestyle='--', linewidth=0.5)

ax3.plot(t_clk0*1e6, v_clk0*1e3, label='0°', linewidth=3, color='tab:blue')
ax3.plot(t_clk180*1e6, v_clk180*1e3, label='180°', linewidth=3, color='tab:orange')
ax3.set_ylabel('mV', fontweight='bold')
ax3.legend(loc='right', fontsize=11)
ax3.grid(which='both', linestyle='--', linewidth=0.5)

ax4.plot(t_clk90*1e6, v_clk90*1e3, label='90°', linewidth=3, color='tab:green')
ax4.plot(t_clk270*1e6, v_clk270*1e3, label='270°', linewidth=3, color='tab:red')
ax4.set_ylabel('mV', fontweight='bold')
ax4.legend(loc='right', fontsize=11)
ax4.grid(which='both', linestyle='--', linewidth=0.5)

ax5.plot(t_upp*1e6, v_upp*1e3, label='UPP', linewidth=3, color='tab:purple')
ax5.plot(t_upn*1e6, v_upn*1e3, label='UPN', linewidth=3, color='tab:brown')
ax5.set_ylabel('mV', fontweight='bold')
ax5.legend(loc='right', fontsize=11)
ax5.grid(which='both', linestyle='--', linewidth=0.5)

ax6.plot(t_dnn*1e6, v_dnn*1e3, label='DNN', linewidth=3, color='tab:pink')
ax6.plot(t_dnp*1e6, v_dnp*1e3, label='DNP', linewidth=3, color='tab:olive')
ax6.set_ylabel('mV', fontweight='bold')
ax6.legend(loc='right', fontsize=11)
ax6.grid(which='both', linestyle='--', linewidth=0.5)

plt.xlim(4.3838, 4.3845) # Focus on a specific time window
fig.supxlabel('Time (us)', fontweight='bold')
plt.subplots_adjust(hspace=-1) # Reduce space between subplots
plt.tight_layout()
plt.savefig('transient_signals_tt_60_1p0V.pdf', dpi=400)
plt.show()


