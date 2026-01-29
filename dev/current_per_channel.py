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
T_avg = 4000.0  # ms
T_total = 60000.0  # 20 segundos
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
exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list = fn.connections(post=post,
                                    w_exc_by_group=w_exc_by_group, inh_syn_params=inh_syn_params, T_total=T_total)
    
# ==================== Configuração dos Medidores ====================
multimeter = nest.Create("multimeter", params={
       "record_from": ["V_m", "g_ex", "g_in"],
      "interval": 0.1  # High temporal resolution
    })
nest.Connect(multimeter, post)

spikedetector = nest.Create("spike_recorder")   
nest.Connect(post, spikedetector)
    
#%%
# ==================== Simulação em Etapas ====================
snapshots = []
titles = ["Before Plasticity", "During Plasticity", "After Plasticity (Detailed Balance)"]

# Snapshot 1: Antes de começar (T=0)
print("Capturing Snapshot 1 (Before)...")
snapshots.append(fn.get_currents_snapshot(post, exc_parrots_list, inh_parrots_list, rates, tau_exc, tau_inh ))

# Simular metade do tempo
print("Simulating first half...")
nest.Simulate(T_total/2)
#%%
# Snapshot 2: Durante (T=10s)
print("Capturing Snapshot 2 (During)...")
snapshots.append(fn.get_currents_snapshot(post, exc_parrots_list, inh_parrots_list, rates, tau_exc, tau_inh))

# Simular restante
print("Simulating second half...")
nest.Simulate(T_total)

# Snapshot 3: Final (T=20s)
print("Capturing Snapshot 3 (After)...")
snapshots.append(fn.get_currents_snapshot(post, exc_parrots_list, inh_parrots_list, rates, tau_exc, tau_inh))

# ==================== Plotagem (Figura 1E) ====================
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True, sharey=True)
channel_idx = np.arange(1, n_groups + 1)

for i, ax in enumerate(axes):
    I_ex, I_in = snapshots[i]
    
    # Excitatory: Círculos pretos fechados
    ax.plot(channel_idx, I_ex, 'ko-', label='Excitatory' if i==0 else "")
    
    # Inhibitory: Círculos brancos (abertos)
    ax.plot(channel_idx, I_in, 'wo-', markeredgecolor='k', label='Inhibitory' if i==0 else "")
    
    ax.set_title(titles[i])
    ax.set_ylabel("Mean Current (pA)")
    
    # Linha guia do canal preferido
    ax.axvline(preferred_group + 1, color='gray', linestyle='--', alpha=0.3)

axes[-1].set_xlabel("Signal Channel #")
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.95))

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(folder, 'figure_1E_current_per_chan.png'), dpi=300)


print("Plotting completed.")


