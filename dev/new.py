#%%
import nest
import numpy as np
import matplotlib.pyplot as plt
import functions_nest as fn
import matplotlib.gridspec as gridspec


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

nest.total_num_virtual_procs=3

# ==================== Parâmetros ====================
T_avg = 2000.0  # ms
T_total = 10000.0  # 20 segundos
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

# Pesos excitatórios fixos (Tuning Curve Inicial)
preferred_group = 4 
w_exc_by_group = fn.exc_weights(n_groups, preferred_group)

exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list, multimeter, spikedetector = fn.connections(post, n_groups, N_exc_per_group, 
                                    N_inh_per_group, w_exc_by_group, inh_syn_params, delay, T_total, dt)
    


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


# Plot Synaptic Currents in Different Time Intervals

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

for ax, (begin, end) in zip(axs, intervals):
    ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(t, I_E/1000, color="black", label="Excitatory")
    ax.plot(t, I_I/1000, color="gray", label="Inhibitory")
    ax.plot(t, I_net/1000, color="green", label="Net")

    ax.set_xlim(begin, end)
    #ax.set_ylim(-20.0,20.0)
    ax.set_ylabel("Synaptic current (nA)")
    ax.legend(frameon=False, loc="upper right")

axs[-1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.show()

#%%
# ==================== Varredura de Taxas Alvo (Figura 1G) ====================

target_rates = [5, 10, 20, 30, 40, 50] # Hz (rho_0)
measured_rates = []

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

#=================================================================================

#%%

output_all_trials = [] # Output spike times all trials
n_trials = 2  # Number of trials
all_input_signals = np.empty((n_groups, n_trials), dtype=object)
correlation_results = []
s_k = np.empty((n_groups, n_trials))  # Input signals PSTH

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

    nest.total_num_virtual_procs=3

    # ==================== Parâmetros ====================

    T_sim = 5000.0  # ms
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

    # Pesos excitatórios fixos (Tuning Curve Inicial)
    preferred_group = 4 
    w_exc_by_group = fn.exc_weights(n_groups, preferred_group)

    exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list, multimeter, spikedetector = fn.connections(post, n_groups, N_exc_per_group, 
                                    N_inh_per_group, w_exc_by_group, inh_syn_params, delay, T_sim, dt)
    
    # ==================== Simulação ====================

    nest.Simulate(T_sim)
    mm = nest.GetStatus(multimeter, "events")[0] #Multimeter data
    spike_n = nest.GetStatus(spikedetector, "events")[0] # Spike detector data post neuron
    output_all_trials.append(spike_n["times"])  # ms

    for g in range(n_groups): 

        spike_input_trial = nest.GetStatus(spike_parrots_list[g], "events")[0]
        spike_times_trial = spike_input_trial["times"]
        all_input_signals[g, n] = spike_times_trial
        


# Análise Pós-Simulação

r_t = fn.get_psth(output_all_trials, n_trials, T_sim, bin_size=5.0)

for g in range(n_groups):

    signals_k = fn.get_psth(all_input_signals[g, :], n_trials, T_sim, bin_size=5.0)
    s_k[g, n] = signals_k

    signal_k = all_input_signals[g,0]

    correlation = fn.calculate_input_output_correlation(r_t, signal_k, bin_size_ms=5.0, dt_signal=0.1)
    correlation_results.append(correlation)

y_pos = np.arange(1, n_groups + 1)
# Plot Raster plot of the fire rate of the input signals

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

# --- Panel 1: Input Rate Heatmap (Raster representation) ---
ax_signal = plt.subplot(gs[0])

# Using imshow to create the color-coded raster
# Aspect='auto' is crucial so the pixels aren't square (time is long)
im = ax_signal.imshow(s_k, 
                      aspect='auto', 
                      cmap='inferno', 
                      interpolation='nearest',
                      extent=[0, T_sim, n_groups + 0.5, 0.5]) # Flip Y to have Ch1 at top

ax_signal.set_xlabel("Time (ms)")
ax_signal.set_ylabel("Input Channel #")
ax_signal.set_title("Input Signal Firing Rates (Hz)")
ax_signal.set_yticks(np.arange(1, n_groups + 1))

# Add colorbar
cbar = plt.colorbar(im, ax=ax_signal, pad=0.02)
cbar.set_label("Rate (Hz)")

# --- Panel 2: Excitatory Weights ---
ax_weights = plt.subplot(gs[1], sharey=ax_signal)

y_pos = np.arange(1, n_groups + 1)
ax_weights.barh(y_pos, w_exc_by_group, color='black', alpha=0.7, height=0.6)

ax_weights.set_xlabel("Weight (nS)")
ax_weights.set_title("Exc. Weights")
ax_weights.grid(True, axis='x', linestyle='--', alpha=0.5)

# Hide Y labels on the right plot since they share the axis
plt.setp(ax_weights.get_yticklabels(), visible=False)
ax_weights.invert_yaxis() # Match the image order (1 at top)

plt.tight_layout()
plt.show()






# %%
