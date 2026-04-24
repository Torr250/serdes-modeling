#Transient plots to generate the spectrum of a signal
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

fosc = 2.5e9  # Oscillation frequency of the PLL in Hz
fref = 150e6  # Reference frequency in Hz

# Read the CSV file with specified column names
csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout2/tran_clk_ss_ff_tt_pex.csv'
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


#plot only the first 100 samples the three signals just to verify they were read correctly
plt.figure(figsize=(10, 6))
plt.plot(t_ff[:100], v_ff[:100], label='Fast-Fast (FF)', color='blue')
plt.plot(t_tt[:100], v_tt[:100], label='Typical-Typical (TT)', color='orange')
plt.plot(t_ss[:100], v_ss[:100], label='Slow-Slow (SS)', color='green')
plt.title('Transient Response of PLL Output for Different Process Corners')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True) 
plt.show()


#Interpolate the three signals to have the same time base
steptime = 1e-12  # 1 ps time step
initial_time = min(t_ff[0], t_tt[0], t_ss[0])
final_time = max(t_ff[-1], t_tt[-1], t_ss[-1])
num_points = int((final_time - initial_time) / steptime) + 1

common_time = np.linspace(initial_time, final_time, num=num_points)
interp_v_ff = np.interp(common_time, t_ff, v_ff)
interp_v_tt = np.interp(common_time, t_tt, v_tt)
interp_v_ss = np.interp(common_time, t_ss, v_ss)


# plot the first 100 samples of the interpolated signals to verify interpolation
plt.figure(figsize=(10, 6))
plt.plot(common_time[:400], interp_v_ff[:400], label='Fast-Fast (FF)', color='blue')
plt.plot(common_time[:400], interp_v_tt[:400], label='Typical-Typical (TT)', color='orange')
plt.plot(common_time[:400], interp_v_ss[:400], label='Slow-Slow (SS)', color='green')
plt.title('Interpolated Transient Response of PLL Output for Different Process Corners')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()

# save the interpolated signals to a new CSV file
interpolated_df = pd.DataFrame({
    'time': common_time,
    'voltage_ff': interp_v_ff,
    'voltage_tt': interp_v_tt,
    'voltage_ss': interp_v_ss
})
interpolated_df.to_csv('interpolated_signals.csv', index=False)


# Print lengths of each new array to verify interpolation
print(f"common_time length: {len(common_time)}")
print(f"interp_v_ff length: {len(interp_v_ff)}")
print(f"interp_v_tt length: {len(interp_v_tt)}")
print(f"interp_v_ss length: {len(interp_v_ss)}")

# I need to generate the eye diagram of the three signals, so I will plot them in a way that they overlap each other, and I will use a time window of 1/fref to plot the eye diagram
eye_window = 1 / fosc  # Time window for one period of the reference frequency
# Plot the eye diagram for the three signals for the first 10 periods of the reference frequency
plt.figure(figsize=(10, 6))
for i in range(10):
    start_time = initial_time + i * eye_window
    end_time = start_time + eye_window
    mask = (common_time >= start_time) & (common_time < end_time)
    plt.plot(common_time[mask] - start_time, interp_v_ff[mask], label='Fast-Fast (FF)', color='blue')
    plt.plot(common_time[mask] - start_time, interp_v_tt[mask], label='Typical-Typical (TT)', color='orange')
    plt.plot(common_time[mask] - start_time, interp_v_ss[mask], label='Slow-Slow (SS)', color='green')
plt.title('Eye Diagram of PLL Output for Different Process Corners')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()

