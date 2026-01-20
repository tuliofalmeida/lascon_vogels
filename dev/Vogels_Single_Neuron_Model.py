#%%
import matplotlib.pyplot as plt
import nest
import numpy as np
import nest.voltage_trace

nest.ResetKernel()

'''Instantiate a leaky integrate-and-fire (IAF) neuron.
   Configure the synapses to be conductance-based.
   Set the reversal potentials to VE​=0 mV for excitation and VI​=−80 mV for inhibition.
   Set the synaptic time constants to τE​=5 ms for excitation and τI​=10 ms for inhibition
'''

# Post Neuron parameters
params_post_neuron = {
    'V_m'       : -60.0, # Membrane potential
   # 'E_L'       : ,  # Leak reversal potential
    'C_m'       : 200,  # Capacity of membrane
    't_ref'     : 5, # Duration of refratory period
    'V_th'      : -50.0,   # spiking threshold
    'V_reset'   : -60.0, # Reset potential of membrane
    'E_ex'      : 0, # Excitatory reversal potential
    'E_in'      : -80.0, # Inhibitory reversal potential
    'g_L'       : 10.0, # Leak conductance
    'tau_syn_ex': 5.0, # Rise time of exc. synaptic alpha function
    'tau_syn_in': 10.0, # Rise time of inh. synaptic alpha function
     'I_e'      : 0  # Constante input current
}


# Vogels plasticity parameters
vogel_params = {
    "tau": 20, # time constant of STDP window
    "eta": 10**(-4), # learning rate
    "alpha": 0.2, # constant depression
    "Wmax": -0.1,  # upper bound (still negative!)
    "weight": -0.3,
    "delay": 1.5,
}

# Create postsynaptic neuron
post_neuron = nest.Create("iaf_cond_exp", 1, params_post_neuron)


# Generate channels s_k signals 
dt = 0.1  # time step in ms
T = 5000  # total time in ms
time = np.arange(0, T, dt)
tau_s = 50 # time constant of the signals

for k in range(8):
    signal_k(time+dt) = epsilon - (epsilon - signal_k(time)) * np.exp(-dt / tau)_s
    signal_k(time+dt) += S_k(time) * w_k
