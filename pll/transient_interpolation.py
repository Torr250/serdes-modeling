#Transient plots to generate the spectrum of a signal
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

fosc = 2.5e9  # Oscillation frequency of the PLL in Hz
fref = 150e6  # Reference frequency in Hz

# Read the CSV file with specified column names
csv_path = 'C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/tran_clk_ss_ff_tt_pex.csv'
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


# Print lengths of each new array to verify interpolation
print(f"common_time length: {len(common_time)}")
print(f"interp_v_ff length: {len(interp_v_ff)}")
print(f"interp_v_tt length: {len(interp_v_tt)}")
print(f"interp_v_ss length: {len(interp_v_ss)}")

# Calculate the FFT of each interpolated signal
fft_v_ff = np.fft.fft(interp_v_ff)
fft_v_tt = np.fft.fft(interp_v_tt)
fft_v_ss = np.fft.fft(interp_v_ss)
freqs = np.fft.fftfreq(len(common_time), d=steptime)
# Calculate the magnitude spectrum
magnitude_ff = np.abs(fft_v_ff)
magnitude_tt = np.abs(fft_v_tt)
magnitude_ss = np.abs(fft_v_ss)
# Only take the positive frequencies
positive_freqs = freqs[freqs >= 0]
magnitude_ff = magnitude_ff[freqs >= 0]
magnitude_tt = magnitude_tt[freqs >= 0]
magnitude_ss = magnitude_ss[freqs >= 0]
# Convert magnitude to dB scale
magnitude_ff = 20 * np.log10(magnitude_ff + 1e-12)  # Adding a small value to avoid log(0)
magnitude_tt = 20 * np.log10(magnitude_tt + 1e-12)  # Adding a small value to avoid log(0)
magnitude_ss = 20 * np.log10(magnitude_ss + 1e-12)  # Adding a small value to avoid log(0)

#Normalize the magnitude spectra
magnitude_ff -= np.max(magnitude_ff)
magnitude_tt -= np.max(magnitude_tt)
magnitude_ss -= np.max(magnitude_ss)

# Plot the magnitude spectrum in dB
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, magnitude_ff, label='Fast-Fast (FF)', color='blue')
plt.plot(positive_freqs, magnitude_tt, label='Typical-Typical (TT)', color='orange')
plt.plot(positive_freqs, magnitude_ss, label='Slow-Slow (SS)', color='green')
plt.title('Magnitude Spectrum of PLL Output for Different Process Corners')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(fosc - fref, fosc + fref)  # Limit x-axis to fosc ± fref
plt.ylim(-100, 0)  # Limit y-axis for better visibility
plt.legend()
plt.grid(True)
plt.show()

# Save the magnitude spectrum data to CSV files
spectrum_data_ff = pd.DataFrame({'Frequency (Hz)': positive_freqs, 'Magnitude (dB)': magnitude_ff})
spectrum_data_tt = pd.DataFrame({'Frequency (Hz)': positive_freqs, 'Magnitude (dB)': magnitude_tt})
spectrum_data_ss = pd.DataFrame({'Frequency (Hz)': positive_freqs, 'Magnitude (dB)': magnitude_ss})
spectrum_data_ff.to_csv('C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/spectrum_ff.csv', index=False)
spectrum_data_tt.to_csv('C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/spectrum_tt.csv', index=False)
spectrum_data_ss.to_csv('C:/Users/adant/Mi unidad (atorralba@inaoe.mx)/Reportes/EDIESCA 2025/PostLayout/spectrum_ss.csv', index=False)
print("Spectrum data saved to spectrum_ff.csv, spectrum_tt.csv, and spectrum_ss.csv")





