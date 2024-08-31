"""
This file shows example of loading in touchstone file for differential channel and generating bode plot and impulse response

To run this file:
    - download zipped touchstone files from : https://www.ieee802.org/3/ck/public/tools/cucable/mellitz_3ck_04_1119_CACR.zip
    - Place the file Tp0_Tp5_28p5db_FQSFP_thru.s4p in the working directory, with this file. It contains s-paramater measurements for 2m copper cable connector with 28dB insertion loss at 26.56Ghz from IEEE 802.df public channels
    - create a subdirectory called 'data' in the working directory. this is where data generated from this script will be saved for use in other examples
"""

#import packages
import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp
from si_prefix import si_format

#datarate
datarate = 8e9
data_period = 1/datarate

#NRZ enconding
nyquist_f = datarate/2
symbol_t = 1/nyquist_f

#simulator step time 1ps
t_d=1e-12

#simulator bit period
samples_per_symbol = round(data_period/t_d)


#load impulse response from MATLAB
ir = sp.io.loadmat('ir_B12.mat')
ir = ir['ir']
ir=ir[:,0]
ir_t =  np.arange(1,len(ir)+1,1)*t_d*1e9

plt.figure(dpi = 1200)
plt.plot(ir_t,ir*1e3)
max_idx = np.where(ir == np.amax(ir))[0][0]
plt.xlim([(max_idx-2*samples_per_symbol)*t_d*1e9,(max_idx+5*samples_per_symbol)*t_d*1e9])
plt.grid()
plt.title("Impulse Response MATLAB")

ch1 = sp.fft.fft(ir)
ch1_20dB = 20*np.log10(np.abs(ch1))
ch1_freqs=(1/(t_d*len(ch1)))*(np.arange(1,len(ir)+1,1))*1e-9

plt.figure(dpi = 1200)
plt.plot(ch1_freqs,ch1_20dB)
ax = plt.gca()
max_idx = np.where(ch1_freqs == 2*nyquist_f*1e-9)[0][0]
ax.set_xlim([0,2*nyquist_f*1e-9])
ax.set_ylim([ch1_20dB[max_idx],0])
plt.grid()
plt.title("Frequency response MATALB IR")


pulse_data = np.zeros(110)
pulse_data[1] = 1

pulse_data_sym = np.repeat(pulse_data,samples_per_symbol)


pulse_response = 0.5*sp.signal.convolve(ir, pulse_data_sym, mode = "full")
plt.figure(dpi = 1200)
pulse_t =  np.arange(1,len(pulse_response)+1,1)*t_d*1e9
plt.plot(pulse_t,pulse_response*1e3)
max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
plt.xlim([(max_idx-2*samples_per_symbol)*t_d*1e9,(max_idx+5*samples_per_symbol)*t_d*1e9])
plt.grid()
plt.title("Pulse response MATLAB IR")

sdp.channel_coefficients(pulse_response, pulse_t*1e-9, samples_per_symbol, 2, 5)


#load in touchstone file
thru_file = "peters_01_0605_B12_thru.s4p"
thru_network = rf.Network(thru_file)

#port definition, is defined in the header of the touchstone file
port_def = np.array([[0, 1],[2, 3]])

#load and source impedance are matched 50 ohms, because charictaristic empedance of the the channel is 50 ohms
Zs = 50
Zl = 50

#compute differential transfer function and impulse response from s-params
H_thru, f, h_thru, t = sdp.four_port_to_diff(thru_network, port_def, Zs, Zl, option = 1, t_d = t_d)

#Plot transfer function of Channel
plt.figure(dpi = 1200)
plt.plot(1e-9*f,20*np.log10(abs(H_thru)), color = "blue", label = "THRU channel", linewidth = 0.8)
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.axvline(x=4,color = 'grey', label = "Nyquist Frequency")
plt.title("Bode Plot SerdesPy")
plt.grid()
ax = plt.gca()
max_idx = np.where(f == 2*nyquist_f)[0][0]
ax.set_xlim([0,2*nyquist_f*1e-9])
ax.set_ylim([20*np.log10(abs(H_thru[max_idx])),0])
plt.legend()

#Plot impulse response from serdespy method, not good
plt.plot(t,h_thru)
plt.title("Impulse Response SerdesPy")


#visualize pulse response using serdespy method, not good

pulse_data = np.ones(samples_per_symbol)
pulse_response = sp.signal.fftconvolve(h_thru, pulse_data, mode = "same")
plt.figure(dpi = 1200)
plt.plot(pulse_response)
sdp.channel_coefficients(pulse_response, t, samples_per_symbol, 2, 5)


pulse_response = 0.5*sp.signal.fftconvolve(ir, pulse_data, mode = "full")
pulse_t =  np.arange(1,len(pulse_response)+1,1)*t_d*1e9
plt.figure(dpi = 1200)
plt.plot(pulse_response)
sdp.channel_coefficients(pulse_response, pulse_t*1e-9, samples_per_symbol, 2, 5)


#crop impulse response and save
#plt.figure(dpi = 1200)
#h_thru_crop = h_thru[0:2000]
#plt.plot(h_thru_crop)



#save pulse response, transfer function, and frequency vector, used in other example files

#save data
np.save("./data/h_thru.npy",ir) #Impulse Response
np.save("./data/f.npy",f) #Frequency 
np.save("./data/TF_thru.npy",H_thru) #Magnitude response
