#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:49:32 2024

@author: atorralba
"""

import serdespy as sdp
import numpy as np
import scipy as sp
import serdes_functions as sdf
from si_prefix import si_format
from prettytable import PrettyTable

def channel_coefficients(pulse_response, samples_per_symbol, n_precursors, n_postcursors):
    #TODO: check for not long enough signal
    #make t optional arg
    """measures and plots channel coefficients from pulse response

    Parameters
    ----------
    pulse_response: array
    
    t: array
        time vector corresponding to pulse response

    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    res : int
        resolution of plot
        
    title: str
        title of plot
    
    Returns
    -------
    channel_coefficients : array
        channel coefficents measured from pulse response

    """
    #number of channel coefficients
    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    
    #find peak of pulse response = main cursor sample time
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    for cursor in range(n_cursors):
        
        #index of channel coefficint
        a = cursor - n_precursors
        
        #measure pulse response
        channel_coefficients[cursor] = pulse_response[max_idx+a*samples_per_symbol]
        
    
    return channel_coefficients



def wc_eyeheight(pulse_response, samples_per_symbol, n_precursors, n_postcursors):
    """measures the worst case eye height using the coefficient method

    Parameters
    ----------
    pulse_response: array
    
    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    Returns
    -------
    wc_eyeheight : int
        worst case eye height in volts

    """
    
    ch1_coeff = channel_coefficients(pulse_response, samples_per_symbol, n_precursors, n_postcursors)

    pulse_peak = ch1_coeff[n_precursors]
    WC0 = 0
    WC1 = 0

    for x in range(n_precursors):
        if ch1_coeff[x] < 0 :
            WC1 = ch1_coeff[x] + WC1;
        else:
            WC0 = ch1_coeff[x] + WC0;
            
    for x in range(n_postcursors):
        if ch1_coeff[n_precursors+1+x] < 0 :
            WC1 = ch1_coeff[n_precursors+1+x] + WC1;
        else:
            WC0 = ch1_coeff[n_precursors+1+x] + WC0;
            
    WCEyeH = 2*(pulse_peak+WC1-WC0)

    #print('WC0 (positive): '+si_format(WC0)+', WC1 (negative): '+si_format(WC1)+', Peak, : '+si_format(pulse_peak))
    #print('WC eye height: '+si_format(WCEyeH))
    
    return WCEyeH
    

def wc_eyeheight_coeff(coeff, samples_per_symbol, n_precursors, n_postcursors):
    """measures the worst case eye height using the coefficient method

    Parameters
    ----------
    coeff: array
        Only channel coefficients
    
    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    Returns
    -------
    wc_eyeheight : int
        worst case eye height in volts

    """
    
    pulse_peak = coeff[n_precursors]
    WC0 = 0
    WC1 = 0

    for x in range(n_precursors):
        if coeff[x] < 0 :
            WC1 = coeff[x] + WC1;
        else:
            WC0 = coeff[x] + WC0;
            
    for x in range(n_postcursors):
        if coeff[n_precursors+1+x] < 0 :
            WC1 = coeff[n_precursors+1+x] + WC1;
        else:
            WC0 = coeff[n_precursors+1+x] + WC0;
            
    WCEyeH = 2*(pulse_peak+WC1-WC0)

    #print('WC0 (positive): '+si_format(WC0)+', WC1 (negative): '+si_format(WC1)+', Peak, : '+si_format(pulse_peak))
    #print('WC eye height: '+si_format(WCEyeH))
    
    return WCEyeH
    

def serdes_evaluation(datarate, ir_channel_file, tx_ffe_taps_list, rx_ctle_gain_list, rx_dfe_taps_list,eyediagram_plot,wc_eyeh_print,pulse_plot, debug_print):
    """Evaluates eye height from the input configuration

    Parameters
    ----------
    datarate: int
        operating datarate
    
    ir_channel_file: string
        channel impulse response file
    
    tx_ffe_taps_list: array, positive values 0 to 0.4
        pre and post cursor FFE coefficients, main cursor is calculated using 
        main = 1 - pre + post, pre and post are sign changed to: -pre, +main, -post
        0: precursor
        1: post cursor
        
    rx_ctle_gain_list: array, positive values 0 to 2 
        DC and AC gain values for the CTLE
        0: DC, low-frequency gain
        1: AC, peaking at nyquist frequency
        
    rx_dfe_taps_list: array, positive and negative values -0.5 to +0.5
        0: tap1
        1: tap2
        2: tap3
        
    eyediagram_plot: string
        Prints the eye diagram of one of all the equalizers
        Enable it uses large amount of memory and cpu
        'not': do not print any eye diagram
        'all': print channel > FFE > CTLE > DFE eye diagrams
        'final': print the eye diagram after all the equalizers
        
    wc_eyeh_print: string
        Prints the WC eye height 
        'not': do not print any WC eye height
        'all': print channel > FFE > CTLE > DFE WC eyes
        'final': print the WC eye after all the equalizers
        
    Returns
    -------
    wc_eyeheight : float
        worst case eye height after all the equalizers
        if the wrong values are introduced output value is -1

    """
    # Known values
    #datarate = 8e9
    #ir_channel_file = 'ir_B20.mat' 
    #tx_ffe_taps_list = [0.000, 0.250] #PCIe P0, Autocalculates main tap, positive values only max value 0.4
    #rx_ctle_gain_list = [6,6] # Positive values only up to 6, in dB
    #rx_dfe_taps_list = [-0.033 , -0.052, -0.015] #Random Taps, positive and negative values up to 0.5
    #eyediagram_plot = 'not' # final, all, not
    #wc_eyeh_print = 'not' #final or all, not
    
    #expected return value 0.3567443163177757

    

    #Verify input data
    for x in range(len(tx_ffe_taps_list)):
        if not( 0 <= tx_ffe_taps_list[x] <= 0.4 ):
            print('Not correct FFE values')  
            return -1
      
    for x in range(len(rx_ctle_gain_list)):
        if not( 0 <= rx_ctle_gain_list[x] <= 6 ):
            print('Not correct CTLE values')
            return -1
    
    for x in range(len(rx_dfe_taps_list)):
        if not( -0.5 < rx_dfe_taps_list[x] < 0.5 ):
            print('Not correct DFE values')    
            return -1
    
    
    #Calculate FFE main tap 
    tx_ffe_main_tap = 1-(tx_ffe_taps_list[0] + tx_ffe_taps_list[1])
    tx_fir_tap_weights = np.array([-tx_ffe_taps_list[0],tx_ffe_main_tap,-tx_ffe_taps_list[1]])
    dfe_tap_weights = np.array(rx_dfe_taps_list) 
    ctle_AdcdB = -rx_ctle_gain_list[0];
    ctle_AacdB = rx_ctle_gain_list[1];
    
    #% Simulator Definitions
    #datarate
    #datarate = 8e9
    data_period = 1/datarate
    
    #NRZ enconding
    nyquist_f = datarate/2
    symbol_t = 1/nyquist_f
    
    #simulator step time 1ps
    t_d=1e-12
    
    #simulator bit period
    samples_per_symbol = round(data_period/t_d)
    
    
    #% Load Impulse Response
    #load impulse response from MATLAB
    ir = sp.io.loadmat(ir_channel_file)
    ir = ir['ir']
    ir=ir[:,0]
    #ir_t =  np.arange(1,len(ir)+1,1)*t_d*1e9
    
    #Frequency response
    ch1 = sp.fft.fft(ir)
    #ch1_20dB = 20*np.log10(np.abs(ch1))
    ch1_freqs=(1/(t_d*len(ch1)))*(np.arange(1,len(ir)+1,1))*1e-9
    
    #Ideal pulse data
    pulse_data = np.zeros(110)
    pulse_data[1] = 1
    
    pulse_data_sym = np.repeat(pulse_data,samples_per_symbol)
    
    #Pulse response only channel
    pulse_response = 0.5*sp.signal.convolve(ir, pulse_data_sym, mode = "full")
    
    
    #%% Generate CTLE 
    #ctle_AdcdB = -6;
    #ctle_AacdB = 6;
    ctle_fz0 = nyquist_f/8;
    ctle_fp2 = 20e9;
    
    ctle_Adc = 10**(ctle_AdcdB/20);
    ctle_Aac = 10**(ctle_AacdB/20);
    ctle_peaking = ctle_Aac/ctle_Adc;
    ctle_fp1 = ctle_peaking*ctle_fz0;
    #ctle_gbw = ctle_Adc*2*np.pi*ctle_fp2;
    
    ctle_wz0 = 2*np.pi*ctle_fz0;
    ctle_wp1 = 2*np.pi*ctle_fp1;
    ctle_wp2 = 2*np.pi*ctle_fp2;
    
    #print('Zero: '+si_format(ctle_fz0)+', P1: '+si_format(ctle_fp1)+', P2, : '+si_format(ctle_fp2))
    #print('AvDC: '+si_format(ctle_AdcdB)+', AvAC: '+si_format(ctle_AacdB)+', Peaking, : '+si_format(ctle_AacdB+ctle_AdcdB))
    
    ctle_n1 = ctle_Adc*ctle_wp1*ctle_wp2;
    ctle_n0 = ctle_Adc*ctle_wp1*ctle_wp2*ctle_wz0;
    ctle_d2 = ctle_wz0;
    ctle_d1 = (ctle_wp1 + ctle_wp2)*ctle_wz0;
    ctle_d0 =  ctle_wp1*ctle_wp2*ctle_wz0;
    
    #frequency vector in rad/s
    f=ch1_freqs*1e9;
    w = f*(2*np.pi)
    
    #calculate Frequency response of CTLE at given frequencies
    w, H_ctle = sp.signal.freqs([ctle_n1, ctle_n0], [ctle_d2, ctle_d1, ctle_d0], w)
    
    #CTLE impulse response
    h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)
    h_ctle = h_ctle[0:1000]
    
    
    #% Generate Equalized pulse responses
    
    #tx_fir_tap_weights = np.array([-0.000, 0.750, -0.250]) #PCIe P0
    #tx_fir_tap_weights = np.array([-0.000, 1.000, -0.000]) #PCIe P4
    
    #Pulse response Channel + FFE
    pulse_response_fir = 0.5*sp.signal.fftconvolve(ir, np.repeat(tx_fir_tap_weights,samples_per_symbol), mode = "full")
    
    #Pulse response Channel + FFE + CTLE
    pulse_response_fir_ctle = sp.signal.fftconvolve(pulse_response_fir, h_ctle, mode = "full")
    
    #DFE definit ions
    max_idx = np.where(pulse_response_fir_ctle == np.amax(pulse_response_fir_ctle))[0][0]
    channel_coefficients = sdf.channel_coefficients(pulse_response_fir_ctle, samples_per_symbol, 3, 3)
    main_cursor = channel_coefficients[3]
    #dfe_tap_weights = channel_coefficients[4:]
    #dfe_tap_weights = np.array([-0.000, 0.000, -0.000])
    dfe_tap_samples = np.repeat(dfe_tap_weights,samples_per_symbol)
    
    pulse_dfe = np.zeros(len(pulse_response_fir_ctle))
    pulse_dfe[(max_idx+round(samples_per_symbol/2)):(max_idx+round(samples_per_symbol/2))+len(dfe_tap_samples)]=dfe_tap_samples
    
    #Pulse response Channel + FFE + CTLE + DFE
    pulse_response_fir_ctle_dfe=pulse_response_fir_ctle + pulse_dfe
    
    #%% Print in console debug info
    if debug_print == 'yes':
        myTable = PrettyTable(["Eq", "Tap1", "Tap2", "Tap3"]) 
          
        # Add rows 
        myTable.add_row(["FFE", str(tx_fir_tap_weights[0]), str(tx_fir_tap_weights[1]), str(tx_fir_tap_weights[2])]) 
        myTable.add_row(["CTLE", "DC: " + str(rx_ctle_gain_list[0]), "AC:" + str(rx_ctle_gain_list[1]),"-"]) 
        myTable.add_row(["DFE", str(dfe_tap_weights[0]), str(dfe_tap_weights[1]), str(dfe_tap_weights[2])]) 
          
        print(myTable)
        
    #%% Pulse response plots
    if pulse_plot == 'all':
        pulse_t =  np.arange(1,len(pulse_response)+1,1)*t_d*1e9
        pulse_fir_t =  np.arange(1,len(pulse_response_fir)+1,1)*t_d*1e9
        pulse_fir_ctle_t =  np.arange(1,len(pulse_response_fir_ctle)+1,1)*t_d*1e9
        pulse_fir_ctle_dfe_t =  np.arange(1,len(pulse_response_fir_ctle_dfe)+1,1)*t_d*1e9
        
        sdp.channel_coefficients(pulse_response, pulse_t*1e-9, samples_per_symbol, 2, 5)
        sdp.channel_coefficients(pulse_response_fir, pulse_fir_t*1e-9, samples_per_symbol, 2, 5)
        sdp.channel_coefficients(pulse_response_fir_ctle, pulse_fir_ctle_t*1e-9, samples_per_symbol, 2, 5)
        sdp.channel_coefficients(pulse_response_fir_ctle_dfe, pulse_fir_ctle_dfe_t*1e-9, samples_per_symbol, 2, 5)
    elif pulse_plot == 'final':
        pulse_fir_ctle_dfe_t =  np.arange(1,len(pulse_response_fir_ctle_dfe)+1,1)*t_d*1e9
        sdp.channel_coefficients(pulse_response_fir_ctle_dfe, pulse_fir_ctle_dfe_t*1e-9, samples_per_symbol, 2, 5)
    
    #%% WC eye height all filters
    
    WCEyeH_ch1 = sdf.wc_eyeheight(pulse_response, samples_per_symbol, 10, 100)
    WCEyeH_ch1_ffe = sdf.wc_eyeheight(pulse_response_fir, samples_per_symbol, 10, 100)
    WCEyeH_ch1_ffe_ctle = sdf.wc_eyeheight(pulse_response_fir_ctle, samples_per_symbol, 10, 100)
    
    #For DFE filter, pre and main pulses are fixed, only post cursors are modified
    pulse_fir_ctle_t =  np.arange(1,len(pulse_response_fir_ctle)+1,1)*t_d*1e9
    #sdp.channel_coefficients(pulse_response_fir_ctle, pulse_fir_ctle_t*1e-9, samples_per_symbol, 2, 5)
    ch1_ffe_ctle_coeff = sdf.channel_coefficients(pulse_response_fir_ctle, samples_per_symbol, 10, 100)
    ch1_ffe_ctle_dfe_coeff = np.zeros(len(ch1_ffe_ctle_coeff))
    
    for x in range(len(dfe_tap_weights)):
        ch1_ffe_ctle_dfe_coeff[x] = dfe_tap_weights[x]
    
    ch1_ffe_ctle_dfe_coeff = np.roll(ch1_ffe_ctle_dfe_coeff,11)
    ch1_ffe_ctle_dfe_coeff = ch1_ffe_ctle_dfe_coeff + ch1_ffe_ctle_coeff
    
    #WCEyeH_ch1_ffe_ctle_dfe = sdf.wc_eyeheight(pulse_response_fir_ctle_dfe, samples_per_symbol, 10, 100)
    WCEyeH_ch1_ffe_ctle_dfe = sdf.wc_eyeheight_coeff(ch1_ffe_ctle_dfe_coeff, samples_per_symbol, 10, 100)
    
    
    if wc_eyeh_print == 'all':
        myTable = PrettyTable([" ","Channel", "FFE", "CTLE", "DFE"])      
        myTable.add_row(["WC Eye Height", si_format(WCEyeH_ch1)+'V', si_format(WCEyeH_ch1_ffe)+'V', si_format(WCEyeH_ch1_ffe_ctle)+'V',si_format(WCEyeH_ch1_ffe_ctle_dfe)+'V'])
        print(myTable)
    elif wc_eyeh_print == 'final':
        myTable = PrettyTable([" ","Ch+FFE+CTLE+DFE"])      
        myTable.add_row(["WC Eye Height",si_format(WCEyeH_ch1_ffe_ctle_dfe)+'V']) 
        print(myTable)
    
    #%% Eye Diagram all filters
    
    if not(eyediagram_plot == 'not'):
        voltage_levels = np.array([-1,1])
        prbs_nbits = 1000
        
        data = np.concatenate((sdp.prbs13(1),sdp.prbs13(1)))[:prbs_nbits+1] #Max 16382
        
        TX = sdp.Transmitter(data, voltage_levels, nyquist_f)
        TX.oversample(samples_per_symbol)
        
        
        max_idx = np.where(ir == np.amax(ir))[0][0]
        signal_out = 0.5*sp.signal.fftconvolve(ir,TX.signal_ideal, mode = "full")[max_idx:max_idx+(prbs_nbits+1)*samples_per_symbol]
        
        TX = sdp.Transmitter(data, voltage_levels, nyquist_f)
        TX.FIR(tx_fir_tap_weights)
        TX.oversample(samples_per_symbol)
        
        max_idx = np.where(ir == np.amax(ir))[0][0]
        signal_out_ffe = 0.5*sp.signal.fftconvolve(ir,TX.signal_ideal, mode = "full")[max_idx:max_idx+(prbs_nbits+1)*samples_per_symbol]
        
        max_idx = np.where(h_ctle == np.amax(h_ctle))[0][0]
        signal_out_ctle_ffe = 1.0*sp.signal.fftconvolve(h_ctle,signal_out_ffe, mode = "full")[max_idx:max_idx+(prbs_nbits+1)*samples_per_symbol]
        
        if eyediagram_plot == 'all':
            RX = sdp.Receiver(signal_out, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)
            sdp.simple_eye(RX.signal, samples_per_symbol*2, np.int16(prbs_nbits/2), TX.UI/TX.samples_per_symbol, "Channel")
            
            RX = sdp.Receiver(signal_out_ffe, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)
            sdp.simple_eye(RX.signal, samples_per_symbol*2, np.int16(prbs_nbits/2), TX.UI/TX.samples_per_symbol, "Channel + FFE")
            
            RX = sdp.Receiver(signal_out_ctle_ffe, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)
            sdp.simple_eye(RX.signal, samples_per_symbol*2, np.int16(prbs_nbits/2), TX.UI/TX.samples_per_symbol, "Channel + FFE + CTLE")
            
            RX = sdp.Receiver(signal_out_ctle_ffe, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)
            RX.nrz_DFE(-dfe_tap_weights)
            sdp.simple_eye(RX.signal, samples_per_symbol*2, np.int16(prbs_nbits/2), TX.UI/TX.samples_per_symbol, "Channel + FFE + CTLE + DFE")
        elif eyediagram_plot == 'final':
            RX = sdp.Receiver(signal_out_ctle_ffe, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)
            RX.nrz_DFE(-dfe_tap_weights)
            sdp.simple_eye(RX.signal, samples_per_symbol*2, np.int16(prbs_nbits/2), TX.UI/TX.samples_per_symbol, "Channel + FFE + CTLE + DFE")
    
    return WCEyeH_ch1_ffe_ctle_dfe
        
    
    