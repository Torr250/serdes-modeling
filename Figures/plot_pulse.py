# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:16:27 2024

@author: adant
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.patches import Rectangle
from scipy.io import loadmat

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


annots = loadmat('Ideal_Pulse.mat')
ideal_pulse_channel = annots['pulse_channel']
ideal_pulse_channel=ideal_pulse_channel[0,:]
ideal_pulse_ui = annots['pulse_ui']
ideal_pulse_ui=ideal_pulse_ui[0,:]


annots = loadmat('B12_Pulse.mat')
b12_pulse_channel = annots['pulse_channel']
b12_pulse_channel=b12_pulse_channel[:,0]
b12_pulse_ui = annots['pulse_ui']
b12_pulse_ui=b12_pulse_ui[0,:]

annots = loadmat('B20_Pulse.mat')
b20_pulse_channel = annots['pulse_channel']
b20_pulse_channel=b20_pulse_channel[:,0]
b20_pulse_ui = annots['pulse_ui']
b20_pulse_ui=b20_pulse_ui[0,:]

annots = loadmat('Molex_Pulse.mat')
MX_pulse_channel = annots['pulse_channel']
MX_pulse_channel=MX_pulse_channel[:,0]
MX_pulse_ui = annots['pulse_ui']
MX_pulse_ui=MX_pulse_ui[0,:]


#annots = loadmat('T20_AC.mat')
#t20_ac_mag = annots['ac_mag']
#t20_ac_mag=t20_ac_mag[0,:]
#t20_ac_freq = annots['ac_freq']
#t20_ac_freq=t20_ac_freq[0,:]



fig, ax = plt.subplots()
#ax.plot(ideal_pulse_ui, ideal_pulse_channel, color = 'black', linewidth = 3) 
ax.plot(b12_pulse_ui, b12_pulse_channel*1e3, color = 'red', linewidth = 3) #-12.138[dB] 
ax.plot(b20_pulse_ui, b20_pulse_channel*1e3, color = 'blue', linewidth = 3) #-15.438[dB]
ax.plot(MX_pulse_ui, MX_pulse_channel*1e3, color = 'green', linewidth = 3) # -19.245[dB] 

#ax.plot(t, c, color = 'red', linewidth = 3)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.grid(True,linestyle='--',which='both')
ax.minorticks_on()

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter('{x:0.0f}')

ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_formatter('{x:1.0f}')

ax.set_xlim([-2, 6])
#ax.set_ylim([-1, 1])


plt.xticks(weight='bold',fontsize=16)
plt.yticks(weight='bold',fontsize=16)

plt.xlabel('Time (UI)',weight='bold',fontsize=16)
plt.ylabel('Voltage (mV)', weight='bold',fontsize=16)

plt.legend(['Nelco4000-13 12" ','Nelco4000-13 20" ', 'FR408 1m (39.3")'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")
#plt.legend(['Input', 'Nelco4000-13 12" ','Nelco4000-13 20" ', 'FR408 1m (39.3")'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")

#ax.text(60, -0.75, 'Test', color='black', weight='bold',fontsize=16, 
        #bbox=dict(facecolor='white', edgecolor='black', pad=5.0)) #round,pad=0.3

#plt.annotate('',xy=(45,-0.45),xytext=(40,-0.75),
             #arrowprops=dict(facecolor='black',width=2,headwidth=10,headlength=10,shrink=0.3))

#ax.add_patch(Rectangle((20, -0.125), 40, 0.25,  edgecolor='black', facecolor="#F8E71C", linewidth=2,linestyle='--'))


#plt.axvline(x = 30, color = 'green', linewidth=2,  linestyle='--')
#plt.axhline(y = -0.95, color = 'green', linewidth=2,  linestyle='--')

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig("plot_pulse.pdf")

plt.show()