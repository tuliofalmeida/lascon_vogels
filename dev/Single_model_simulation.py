# %%
import nest
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import functions_nest as fn
nest.ResetKernel()
resolution = 0.1  # ms
nest.SetKernelStatus({
    "resolution": resolution,   
    "print_time": True
})
np.random.seed(42)

# %%
"""
# Input Architecture 
"""

# %%
# ==================== Neuron Parameters =================
neuron_params = {
    "C_m": 200.0,        # pF (equivale a tau=20 ms con gL=10 nS)
    "g_L": 10.0,         # nS
    "E_L": -60.0,        # mV
    "V_th": -50.0,       # mV
    "V_reset": -60.0,    # mV
    "t_ref": 5.0,        # ms
    "E_ex": 0.0,         # mV
    "E_in": -80.0        # mV
}

# Synapse time constants
tau_exc = 5.0    # ms (τ_E = 5 ms)
tau_inh = 10.0   # ms (τ_I = 10 ms)

# Channel Parameters
n_groups = 8                    
N_exc_per_group = 100           
N_inh_per_group = 25  

delay = 1.5  # ms
# ==================== Create Signal Modulation ====================

T = 20000.0  # ms
dt = 0.1  # ms
tau = 50.0  # ms
time = np.arange(0, T, dt)

alpha = np.exp(-dt / tau)
eps = np.random.uniform(-0.5, 0.5, size=len(time))
s = np.zeros_like(time)
rates = []

for g in range(n_groups):
    s = np.zeros_like(time)
    # Generate filtered noise signal
    for t in range(1, len(time)):
        s[t] = eps[t]- (eps[t]-s[t-1])*alpha

    # Rectify and normalize signal
    s[s<0] = 0
    s *= (500 * dt) / np.max(s)   
        
    mask = np.random.rand(len(s)) < 0.5
    s *= mask

    # Convert to rate (Hz)
    rate = 5.0 + s / dt  # spikes per second (Hz)

    rates.append(rate.copy())


# %%
# ==================== Create Post Neuron ====================

post = fn.post_neuron("iaf_cond_alpha", tau_exc, tau_inh, n=1, neuron_params=neuron_params)


# ==================== Connection Parameters  ====================

# Parameters for inhibitory synapses (Vogels-Sprekeler)

inh_syn_params = {
    "synapse_model": "vogels_sprekeler_synapse",
    "weight": -0.1,      # Peso inicial DÉBIL (importante)
    "delay": delay,
    "eta": 0.001,        # Tasa de aprendizaje
    "alpha": 0.2,      # Ratio
    "tau": 20.0,        # ms (constante de tiempo STDP)
    "Wmax": -100.0,      # Peso máximo
}

# Parameters for excitatory synapses (fixed weights)

preferred_group = 4  # Grupo 5 (0-based index 4)
w_exc_max = 3.0   # nS para grupo preferido
w_exc_min = 1.5     # nS para grupos no preferidos

w_exc_by_group = fn.exc_weights(n_groups, preferred_group)


# ==================== Create Connections ====================

for g in range(n_groups):


    # Poisson generator

     gen = nest.Create("inhomogeneous_poisson_generator")

     t_now = nest.GetKernelStatus("biological_time")
     dt_sim = nest.GetKernelStatus()["resolution"]
     rate_times = t_now + dt_sim + time

     print("kernel time:", t_now)
     print("first rate time:", rate_times[0])
     print("difference:", rate_times[0] - t_now)

     
     nest.SetStatus(gen, params={
    "rate_times": rate_times,
    "rate_values": rates[g],
    "allow_offgrid_times": True})

    # Excitatory connections
     exc_parrots = nest.Create("parrot_neuron", N_exc_per_group)

     nest.Connect(gen, exc_parrots, conn_spec="all_to_all")
     nest.Connect(exc_parrots, post, conn_spec="all_to_all",
     syn_spec={
        "synapse_model": "static_synapse",
        "weight": w_exc_by_group[g],
        "delay": delay,
       # "receptor_type": 1 # Excitatory receptor
     })

   # Inhibitory connections
     inh_parrots = nest.Create("parrot_neuron", N_inh_per_group)

     nest.Connect(gen, inh_parrots, conn_spec="all_to_all")
     nest.Connect(inh_parrots, post, conn_spec="all_to_all",
     syn_spec={
        **inh_syn_params,
       # "receptor_type": 2 # Inhibitory receptor
     })


# %%
#==================== Simulate and Record Data ====================

multimeter = nest.Create("multimeter", params={
    "record_from": ["V_m", "g_ex", "g_in"],
    "interval": 0.1  # High temporal resolution
})
nest.Connect(multimeter, post)

nest.Simulate(T) 
events = nest.GetStatus(multimeter, "events")[0] #Multimeter data

t = events["times"]  # ms
V = events["V_m"]   # mV
gE = events["g_ex"]  #nS
gI = events["g_in"] #nS


E_ex = neuron_params["E_ex"] # mV
E_in = neuron_params["E_in"] # mV

I_E = -gE * (-E_ex + V) # pA (since nS × mV = pA)
I_I = -gI * (- E_in + V)  # pA (since nS × mV = pA)
I_net = I_E + I_I  # pA

# %%
#==================== Ploting Data ====================

T_avg = 2000.0  # ms 

t_begin = (0.0, T_avg)

t_mid_center = T / 2
t_mid = (
    t_mid_center - T_avg / 2,
    t_mid_center + T_avg / 2
)

t_end = (T - T_avg, T)

intervals = [t_begin, t_mid, t_end]


# Plot Synaptic Currents in Different Time Intervals

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

for ax, (begin, end) in zip(axs, intervals):
    ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(t, I_E/1000, color="black", label="Excitatory")
    ax.plot(t, I_I/1000, color="gray", label="Inhibitory")
    ax.plot(t, I_net/1000, color="green", label="Net")

    ax.set_xlim(begin, end)
    ax.set_ylim(-20.0,20.0)
    ax.set_ylabel("Synaptic current (nA)")
    ax.legend(frameon=False, loc="upper right")
    
 
    #ax.set_title(f"Intervalo: {inicio} - {fin} ms")

axs[-1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.show()


