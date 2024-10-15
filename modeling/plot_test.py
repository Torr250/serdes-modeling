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

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

t = np.arange(0.0, 100.0, 0.1)
s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)
c = np.cos(0.1 * np.pi * t) * np.exp(-t * 0.01)

fig, ax = plt.subplots()
ax.plot(t, s, color = 'blue', linewidth = 3)
ax.plot(t, c, color = 'red', linewidth = 3)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.grid(True,linestyle='--',which='both')
ax.minorticks_on()

ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter('{x:0.0f}')

ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_formatter('{x:1.2f}')

ax.set_xlim([20, 80])
ax.set_ylim([-1, 1])


plt.xticks(weight='bold',fontsize=16)
plt.yticks(weight='bold',fontsize=16)

plt.xlabel('Time (ps)',weight='bold',fontsize=16)
plt.ylabel('Amplitude (mV)', weight='bold',fontsize=16)

plt.legend(['Sin','Cos'],prop = { "size": 16 ,'weight':'bold'}, loc ="upper right")

ax.text(60, -0.75, 'Test', color='black', weight='bold',fontsize=16, 
        bbox=dict(facecolor='white', edgecolor='black', pad=5.0)) #round,pad=0.3

plt.annotate('',xy=(45,-0.45),xytext=(40,-0.75),
             arrowprops=dict(facecolor='black',width=2,headwidth=10,headlength=10,shrink=0.3))

ax.add_patch(Rectangle((20, -0.125), 40, 0.25,  edgecolor='black', facecolor="#F8E71C", linewidth=2,linestyle='--'))


plt.axvline(x = 30, color = 'green', linewidth=2,  linestyle='--')
plt.axhline(y = -0.95, color = 'green', linewidth=2,  linestyle='--')


plt.show()