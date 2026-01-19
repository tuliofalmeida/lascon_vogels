'''Instantiate a leaky integrate-and-fire (IAF) neuron.
   Configure the synapses to be conductance-based.
   Set the reversal potentials to VE​=0 mV for excitation and VI​=−80 mV for inhibition.
   Set the synaptic time constants to τE​=5 ms for excitation and τI​=10 ms for inhibition
'''

import matplotlib.pyplot as plt
import nest
import numpy as np

nest.ResetKernel()

# Parameters

n_groups = 8
n_exc = 100
n_inh = 25

# Neuron parameters
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
    'tau_syn_ex': 5, # Rise time of exc. synaptic alpha function
    'tau_syn_in': 10, # Rise time of inh. synaptic alpha function
     'I_e'    : 0  # Constante input current
}

# Neuron models
post_model = 'iaf_cond_exp'
pre_model = 'poisson_generator' 

# Excitatory synapse parameters
exc_weight = 1.0
exc_delay = 1.5

# Inhibitory plastic synapse parameters
vogel_params = {
    "tau": 20, # time constant of STDP window
    "eta": 10**(-4), # learning rate
    "alpha": 0.2, # constant depression
    "Wmax": 0.0,  # upper bound (still negative!)
    "weight": 0.1
}


# Create postsynaptic neuron
post_neuron = nest.Create(post_model, params_post_neuron)


# Create input groups
exc_groups = []
inh_groups = []

for g in range(n_groups):
    
    exc = nest.Create(pre_model, n_exc, params = {'rate': 8000.0})
    inh = nest.Create(pre_model, n_inh, params = {'rate': 8000.0})

    exc_groups.append(exc)
    inh_groups.append(inh)

nest.CopyModel(
    "static_synapse",
    "excitatory_static",
    {
        "weight": exc_weight,
        "delay": exc_delay
    }
)

for exc in exc_groups:
    nest.Connect(
        exc,
        post_neuron,
        syn_spec = 'excitatory_static'
    )

nest.CopyModel(
    "vogels_sprekeler_synapse",
    "inhibitory_plastic",
    vogel_params
)

for inh in inh_groups:
    nest.Connect(
        inh,
        post_neuron,
        syn_spec="inhibitory_plastic"
    )

sd = nest.Create('spike_recorder')
nest.Connect(post_neuron, sd)

nest.Simulate(1000.0)

print("Number of spikes:", nest.GetStatus(sd, "n_events")[0])




'''Use the Vogels plasticity already in NEST
Extract the parameters from the paper and create a file/table with it'''


'''Create 1,000 input synapses directed at a single postsynaptic cell.

Divide these inputs into 8 independent groups representing signal channels.
Generate it with NEST seed method, if the code takes too long to run, save this arrays to be loaded
Composition per group: 100 excitatory synapses and 25 inhibitory synapses.
Input Signals: Generate temporally modulated rate signals (time constant τ≈50 ms) to mimic ongoing sensory activity.
Spike Generation: Convert these rate signals into independent Poisson spike trains (125 distinct trains per signal group)'''

