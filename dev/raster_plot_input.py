#%%
import nest
import numpy as np
import matplotlib.pyplot as plt
import functions_nest as fn
import matplotlib.gridspec as gridspec

import os
import matplotlib.pyplot as plt

folder = 'fig'
if not os.path.exists(folder):
    os.makedirs(folder)


#%%
# Configuração Inicial
nest.ResetKernel()
resolution = 0.1
nest.SetKernelStatus({
    "resolution": resolution,
    "print_time": True,
    "local_num_threads": 1
})
np.random.seed(42)

#nest.total_num_virtual_procs=3

# ==================== Parâmetros ====================
T_avg = 2000.0  # ms
T_sim = 40000.0  # 20 segundos
dt = 0.1  # ms


neuron_params = {
    "C_m": 200.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "V_th": -50.0,
    "V_reset": -60.0,
    "t_ref": 5.0,
    "E_ex": 0.0,
    "E_in": -80.0
}

tau_exc = 5.0
tau_inh = 10.0
n_groups = 8
N_exc_per_group = 100
N_inh_per_group = 25
delay = 1.5

output_all_trials = [] # Output spike times all trials
n_trials = 4  # Number of trials
all_input_signals = np.empty((n_groups, n_trials), dtype=object)  # Input signals all trials
correlation_results = []
s_k = np.empty((n_groups, n_trials), dtype=object)  # Input signals PSTH

#%%
# ==================== Construção da Rede ====================
post = fn.post_neuron("iaf_cond_alpha", tau_exc, tau_inh, n=1, neuron_params=neuron_params)
tau_stdp = 20.0
inh_syn_params = {
    "synapse_model": "vogels_sprekeler_synapse",
    "weight": -0.1,      # Inicialmente fraco (Before)
    "delay": delay,
    "eta": 0.001,       
    "alpha": 0.2,
    "tau": tau_stdp,
    "Wmax": -100.0,
}
#%%
# Pesos excitatórios fixos (Tuning Curve Inicial)
preferred_group = 4 
w_exc_by_group = fn.exc_weights(n_groups, preferred_group)

#%%
for n in range(n_trials):  # 3 trials

    print(f"Trial {n+1}/3")
    nest.ResetKernel()
    resolution = 0.1
    nest.SetKernelStatus({
        "resolution": resolution,
        "print_time": True,
        "local_num_threads": 1
    })
    np.random.seed(42)

    #nest.total_num_virtual_procs=3


    # ==================== Construção da Rede ====================
    post = fn.post_neuron("iaf_cond_alpha", tau_exc, tau_inh, n=1, neuron_params=neuron_params)

    exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list = fn.connections(post=post,
                                    w_exc_by_group=w_exc_by_group, inh_syn_params=inh_syn_params, T_total=T_sim)
    
    # ==================== Configuração dos Medidores ====================
    multimeter = nest.Create("multimeter", params={
       "record_from": ["V_m", "g_ex", "g_in"],
      "interval": 0.1  # High temporal resolution
    })
    nest.Connect(multimeter, post)

    spikedetector = nest.Create("spike_recorder")   
    nest.Connect(post, spikedetector)
    
    # ==================== Simulação ====================

    nest.Simulate(T_sim)
    mm = nest.GetStatus(multimeter, "events")[0] #Multimeter data
    spike_n = nest.GetStatus(spikedetector, "events")[0] # Spike detector data post neuron
    output_all_trials.append(spike_n["times"])  # ms

    for g in range(n_groups): 

        spike_input_trial = nest.GetStatus(spike_parrots_list[g], "events")[0]
        spike_times_trial = spike_input_trial["times"]
        all_input_signals[g, n] = spike_times_trial
        

#%%
# Análise Pós-Simulação

r_t = fn.get_psth(output_all_trials, n_trials, T_sim, bin_size_ms=5.0)
print("PSTH of output neuron calculated.")

for g in range(n_groups):

    g_trials = all_input_signals[g, :]

    all_spikes_flat = np.concatenate(g_trials)    
    signals_k = fn.get_psth(all_spikes_flat, n_trials, T_sim, bin_size_ms=5.0)
    s_k[g, n] = signals_k

    signal_k = all_input_signals[g,0]

    correlation = fn.calculate_input_output_correlation(r_t, signal_k, bin_size_ms=5.0, dt_signal=0.1)
    correlation_results.append(correlation)

print("Input-output correlations calculated: ", correlation_results)

# Plotting Input Signals and Weights
print("Plotting Input Signals and Weights...")

fn.plot_input_raster_and_weights(signals =  all_input_signals, weights_list=w_exc_by_group, dt_sim=0.1, bin_size_ms=5.0)