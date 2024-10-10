#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:19:10 2024

@author: atorralba

Pre-requisites
SerdesPy
    Python 3.7+, evaluated in 3.11
    pip install serdespy
    
si_prefix
    pip install si-prefix
"""
import matplotlib.pyplot as plt
import serdes_functions as sdf
from si_prefix import si_format




datarate = 8e9
ir_channel_file = 'ir_T20.mat'  # ir_B12, ir_B20, ir_T20
# If we set all the values to 0, the channel is not equalized
tx_ffe_taps_list = [0.1, 0.18] #PCIe P0 [0.000, 0.250], Autocalculates main tap, positive values only max value 0.4
rx_ctle_gain_list = [1.3,2.1] # CTLE -6dB and 6dB [6,6],Positive values from 0 up to 6
rx_dfe_taps_list = [-0.04 , 0.0, 0.0] #Random Taps [0.033 , 0.052, 0.015] , positive and negative values up to 0.5
eyediagram_plot = 'all' # final, all, not
wc_eyeh_print = 'all' #final, all, not
pulse_plot = 'all' #final, all, not
debug_print = 'yes'

wceye = sdf.serdes_evaluation(datarate, ir_channel_file, tx_ffe_taps_list, rx_ctle_gain_list, rx_dfe_taps_list, eyediagram_plot, wc_eyeh_print,pulse_plot,debug_print)

print('WC eye height: '+si_format(wceye)+'V')

plt.show()
