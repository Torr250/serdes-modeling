# serdes-modeling
 Serdes Modeling


 Two folders evaluation and modeling.

 Modeling
=========
 Model each element of a SerDes system: Channel, Feed Forward Equalizer(FFE), Continuous Time Linear Equalizer (CTLE), Decision Feedback Equalizer (DFE).
 TODO: add a description of each element 
 
 Evaluation
=========
Evaluates Worst Case Eye Height for a SerDes system based on the channel, equalization coefficients and datarate.

Descipcion de la funcion

```
serdes_evaluation(datarate, ir_channel_file, tx_ffe_taps_list, rx_ctle_gain_list, rx_dfe_taps_list, eyediagram_plot, wc_eyeh_print)

datarate: velocidad de transmision
ir_channel_file: respuesta al impulso del canal, archivo .mat
tx_ffe_taps_list = lista de coeficientes para un equalizador de 3 taps [pre, main, post] donde main = 1 - pre + post, main se calcula automaticamente
rx_ctle_gain_list = Ganancia de baja y alta frecuencia del CTLE en voltaje [Adc,Aac] 
rx_dfe_taps_list = Lista de coeficientes del filtro DFE de 3 taps [tap1, tap2 tap3]
eyediagram_plot = visualiza el diagrama de ojo de 1000 muestras aleatorias. "all" todos los filtros, "final" respuesta final, "not" ninguno
wc_eyeh_print = imprime el resultado del ojo, "all" todos los filtros, "final" respuesta final, "not" ninguno
```

 Ejemplo
=========
```
import serdes_functions as sdf
from si_prefix import si_format

datarate = 8e9
ir_channel_file = 'ir_B20.mat'  # ir_B12, ir_B20, ir_T20
# If we set all the values to 0, the channel is not equalized
tx_ffe_taps_list = [0.000, 0.250] #PCIe P0 main = 0.75, pre = 0, post = 0.25, entrada [pre,post], rango 0 a 0.4
rx_ctle_gain_list = [2,2] # Ganancia Adc y Aac, entrada [Adc, Aac], rango 0 a 2
rx_dfe_taps_list = [-0.033 , -0.052, -0.015] #Taps del DFE [tap1,tap2, tap3] rango -0.5 a 0.5
eyediagram_plot = 'not' # final, all, not
wc_eyeh_print = 'not' #final, all, not

wceye = sdf.serdes_evaluation(datarate, ir_channel_file, tx_ffe_taps_list, rx_ctle_gain_list, rx_dfe_taps_list, eyediagram_plot, wc_eyeh_print)

print('WC eye height: '+si_format(wceye)+'V')

>>>WC eye height: 356.7 mV

```

 Dependencies
=========
Need to add serdes-py to process some data and si_prefix to print the eye height in scientific notation 
```
pip install serdespy
pip install si-prefix
```


Working on pulse response plots
