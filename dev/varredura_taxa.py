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
# ==================== Simulação ====================

nest.Simulate(T_total)


#%%
#==================== Ploting Data ====================
 
mm = nest.GetStatus(multimeter, "events")[0] #Multimeter data

spike = nest.GetStatus(spikedetector, "events")[0] # Spike detector data

t = mm["times"]  # ms
V = mm["V_m"]   # mV
gE = mm["g_ex"]  #nS
gI = mm["g_in"] #nS

t_sp = spike["times"]  # ms
V_sp = spike["senders"]  # Spike senders


E_ex = neuron_params["E_ex"] # mV
E_in = neuron_params["E_in"] # mV

I_E = gE * (-E_ex + V) # pA (since nS × mV = pA)
I_I = gI * (- E_in + V)  # pA (since nS × mV = pA)
I_net = I_E + I_I  # pA

# Define Time Intervals for Plotting
 
t_begin = (0.0, T_avg)

t_mid_center = T_total / 2
t_mid = (
    t_mid_center - T_avg / 2,
    t_mid_center + T_avg / 2
)

t_end = (T_total - T_avg, T_total)

intervals = [t_begin, t_mid, t_end]



#%%
# ==================== Varredura de Taxas Alvo (Figura 1G) ====================

target_rates = [5, 10, 20, 30, 40, 50] # Hz (rho_0)
measured_rates = []
T_sim = 15000.0  # ms

print("Iniciando varredura de taxas alvo (Figura 1G)...")
print(f"{'Alvo (Hz)':<10} | {'Medido (Hz)':<10}")
print("-" * 25)

for rho in target_rates:
    rate = fn.run_simulation_for_target(rho)
    measured_rates.append(rate)
    print(f"{rho:<10.1f} | {rate:<10.1f}")

    plt.figure(figsize=(6, 6))

# Linha de identidade (Target ideal)
plt.plot([0, 60], [0, 60], 'k--', label="Identity (Ideal)", alpha=0.5)

# Dados simulados
plt.plot(target_rates, measured_rates, 'ko-', markersize=8, label="Simulation", linewidth=1.5)

plt.xlabel(r"Target Rate $\rho_0$ (Hz)")
plt.ylabel("Output Firing Rate (Hz)")
plt.title("Figure 1G Reproduction: Target vs Output Rate")
plt.legend(frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis([0, 60, 0, 60])
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(folder, 'figure_1G_target_vs_output_rate.png'), dpi=300)


print("Varredura de taxas alvo concluída.")
#=================================================================================
