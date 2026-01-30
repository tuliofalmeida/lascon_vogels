
"""Functions to create neurons and weights in NEST."""

import nest
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.gridspec as gridspec
# ==================== Parâmetros Globais ====================
nest.ResetKernel()
resolution = 0.1
nest.SetKernelStatus({
    "resolution": resolution,   
    "print_time": True
})
np.random.seed(42)



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

def post_neuron(type: str, tau_exc : float, tau_inh : float, n: int, neuron_params=neuron_params):

    """Create postsynaptic neurons of a given type and number."""
    nest.SetDefaults(type, neuron_params)
    if type == "iaf_cond_alpha":
        neuron = nest.Create("iaf_cond_alpha", params={
            **neuron_params,
            "tau_syn_ex": tau_exc,
            "tau_syn_in": tau_inh
        }, n=n)
    elif type == "aeif_cond_exp":
        neuron = nest.Create("aeif_cond_exp", params={
            **neuron_params,
            "tau_syn_ex": tau_exc,
            "tau_syn_in": tau_inh
        }, n=n)
    else:
        raise ValueError("Unknown neuron type")
    return neuron



def exc_weights(n_groups: int, P: int):
    """Generate random excitatory weights for a given number of groups."""
    w_exc_by_group = []
    for g in range(n_groups):
        weight = np.random.uniform(0, 0.1) + 0.3 + 1.1/(1+np.power(g+1 - P, 4))
        w_exc_by_group.append(weight)

    return w_exc_by_group


# ==================== Função de Cálculo de Corrente ====================
def get_currents_snapshot(post_node, exc_sources, inh_sources, rates_list, tau_exc, tau_inh, V_hold=-60.0):
    """
    Calcula a corrente média teórica baseada nos pesos atuais.
    """
    I_ex_vals = []
    I_in_vals = []
    
    for g in range(len(rates_list)):
        # 1. Obter soma dos pesos Excitatórios
        conns_ex = nest.GetConnections(source=exc_sources[g], target=post_node)
        w_ex_sum = np.sum(nest.GetStatus(conns_ex, 'weight')) # nS
        
        # 2. Obter soma dos pesos Inibitórios (que mudam com o tempo)
        conns_in = nest.GetConnections(source=inh_sources[g], target=post_node)
        w_in_sum = np.sum(nest.GetStatus(conns_in, 'weight')) # nS
        
        # 3. Taxa média do canal
        avg_rate = np.mean(rates_list[g]) # Hz
        
        # 4. Condutância Média (nS) = W_sum * Rate * Tau * (1/1000)
        G_ex = w_ex_sum * avg_rate * tau_exc * 1e-3
        G_in = w_in_sum * avg_rate * tau_inh * 1e-3
        
        # 5. Corrente (pA) = G * (V - E)
        # Usamos valores absolutos para plotar magnitudes como na Figura 1E
        curr_ex = abs(G_ex * (V_hold - neuron_params['E_ex']))
        curr_in = abs(G_in * (V_hold - neuron_params['E_in']))
        
        I_ex_vals.append(curr_ex)
        I_in_vals.append(curr_in)
        
    return I_ex_vals, I_in_vals


def connections(post, w_exc_by_group, inh_syn_params, T_total, n_groups=8, N_exc_per_group=100, N_inh_per_group=25,
                delay=1, dt=0.1):
    
    """Create connections from inhomogeneous Poisson generators to the postsynaptic neuron."""

    exc_parrots_list = []
    inh_parrots_list = []
    spike_parrots_list = []
    time = np.arange(0, T_total, dt)
    tau_noise = 50.0
    alpha_noise = np.exp(-dt / tau_noise)
    rates = []

    for g in range(n_groups):
        eps = np.random.uniform(-0.5, 0.5, size=len(time))
        s = np.zeros_like(time)
        for t_idx in range(1, len(time)):
            s[t_idx] = eps[t_idx] - (eps[t_idx] - s[t_idx-1]) * alpha_noise
        
        s[s<0] = 0
        s_max = np.max(s)
        if s_max > 0:
            s *= (500 * dt) / s_max
        
        mask = np.random.rand(len(s)) < 0.5
        s *= mask
        rate = 5.0 + s / dt 
        rates.append(rate) # Guardamos a array completa
        

    # Criar conexões
    for g in range(n_groups):
        gen = nest.Create("inhomogeneous_poisson_generator")
        
        t_now = nest.GetKernelStatus("biological_time")
        dt_sim = nest.GetKernelStatus()["resolution"]
        rate_times = t_now + dt_sim + time
        
        nest.SetStatus(gen, params={
            "rate_times": rate_times,
            "rate_values": rates[g],
            "allow_offgrid_times": True
        })

        # Excitatory
        exc_parrots = nest.Create("parrot_neuron", N_exc_per_group)
        exc_parrots_list.append(exc_parrots)
        nest.Connect(gen, exc_parrots, conn_spec="all_to_all")
        nest.Connect(exc_parrots, post, conn_spec="all_to_all",
                    syn_spec={
                        "synapse_model": "static_synapse",
                        "weight": w_exc_by_group[g], "delay": delay})
        spike_parrots = nest.Create("spike_recorder")
        nest.Connect(exc_parrots[0], spike_parrots)

        # Inhibitory
        inh_parrots = nest.Create("parrot_neuron", N_inh_per_group)
        inh_parrots_list.append(inh_parrots)
        nest.Connect(gen, inh_parrots, conn_spec="all_to_all")
        nest.Connect(inh_parrots, post, conn_spec="all_to_all",
                    syn_spec={**inh_syn_params})
        
    
        spike_parrots_list.append(spike_parrots)

    return exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list


def run_simulation_for_target(target_rate_hz, tau_stdp=20.0, N_inh_per_group=25, N_exc_per_group=100, n_groups=8, T_sim=15000.0):
    """
    Configura e roda a simulação para uma taxa alvo específica.
    Retorna a taxa de disparo média final do neurônio.
    """
    nest.ResetKernel()
    resolution = 0.1
    nest.SetKernelStatus({
        "resolution": resolution,
        "local_num_threads": 1,
        "print_time": True 
    })
 
    
    # CÁLCULO DO ALPHA COM BASE NA TAXA ALVO
    # Alpha = 2 * tau (s) * rho (Hz)
    alpha_val = 2 * (tau_stdp / 1000.0) * target_rate_hz
    
    inh_syn_params = {
        "synapse_model": "vogels_sprekeler_synapse",
        "weight": -0.1,      
        "eta": 0.005,        # Taxa de aprendizado acelerada para o teste
        "tau": tau_stdp,
        "alpha": alpha_val,  # <--- Aqui definimos o alvo
        "delay": 1.0,
        "Wmax": -100.0
    }


    # ==================== Construção da Rede ====================

    post=post_neuron("iaf_cond_alpha", tau_exc=5.0, tau_inh=10.0, n=1, neuron_params=neuron_params)
    exc_parrots_list, inh_parrots_list, time, rates, spike_parrots_list  = connections(
        post=post,
        n_groups=n_groups,
        N_exc_per_group=N_exc_per_group,
        N_inh_per_group=N_inh_per_group,
        w_exc_by_group=exc_weights(n_groups, P=4),
        inh_syn_params=inh_syn_params,
        delay=1.0,
        T_total=T_sim,
        dt=0.1
    )
    
    # =================== Recorder ====================
    spikedetector = nest.Create("spike_recorder")
    nest.Connect(post, spikedetector)
    

    # ==================== Simulação ====================
    nest.Simulate(T_sim)
    
    # ==================== Análise ====================
    # Calcular taxa apenas nos últimos 5 segundos (estado estacionário)
    events = nest.GetStatus(spikedetector, "events")[0]
    times = events["times"]
    
    t_start_measure = T_sim - 5000.0
    count = np.sum(times > t_start_measure)
    rate_measured = count / 5.0 # dividido por 5 segundos

    inputs = []

    # Calcular taxa de entrada média
    for g in range(n_groups):
        input = nest.GetStatus(spike_parrots_list[g], "events")[0]
        parrot_spike_times = input["times"]
        count = np.sum(parrot_spike_times > t_start_measure)
        rate_g = count / 5.0 # dividido por 5 segundos
        inputs.append(rate_g)
    
    rate_input = np.mean(inputs)
    
    return rate_measured, rate_input


def get_psth(all_spike_times, n_trials, t_sim, bin_size_ms=5.0):
    """
    Converts raw spike times from multiple trials into a firing rate histogram (PSTH).
    
    Args:
        all_spike_times (list or array): A flat list/array containing all spike times 
                                         from all trials combined.
        n_trials (int): Total number of trials run.
        t_sim (float): Total simulation duration in ms.
        bin_size_ms (float): Width of the time bin in ms (default 5.0ms).
        
    Returns:
        r_t (np.array): Array of firing rates (Hz) per bin.
        time_bins (np.array): The time centers of the bins.
    """
    # Create bin edges
    bins = np.arange(0, t_sim + bin_size_ms, bin_size_ms)
    
    # Compute histogram
    counts, _ = np.histogram(all_spike_times, bins=bins)
    
    # Convert counts to Rate (Hz)
    # Rate = Count / (Number of Trials * Bin Duration in seconds)
    bin_duration_sec = bin_size_ms / 1000.0
    r_t = counts / (n_trials * bin_duration_sec)
    
    return r_t

def calculate_input_output_correlation(r_t, signal_k, bin_size_ms, dt_signal):
    """
    Calculates the Pearson correlation between the Output Rate r(t) 
    and a specific Input Signal s_k(t).
    
    Matches the definition: <(s-<s>)(r-<r>)> / (std(s)*std(r))
    
    Args:
        r_t (np.array): The binned output firing rate (from get_psth).
        signal_k (np.array): The high-resolution input signal array.
        bin_size_ms (float): The bin size used for r_t (e.g., 5.0).
        dt_signal (float): The time step of the input signal (e.g., 0.1).
        
    Returns:
        float: Pearson correlation coefficient.
    """
    # 1. Downsample the Input Signal to match the PSTH bin size
    # We calculate the mean of the signal within each 5ms bin
    factor = int(bin_size_ms / dt_signal)
    
    # Ensure the signal length is divisible by the factor for reshaping
    # Truncate slightly if necessary (e.g. if signal is 1 step longer than bins)
    limit = len(r_t) * factor
    if len(signal_k) > limit:
        signal_k = signal_k[:limit]
    elif len(signal_k) < limit:
        # If signal is shorter (rare), trim r_t
        limit = len(signal_k) // factor * factor
        signal_k = signal_k[:limit]
        r_t = r_t[:len(signal_k)//factor]

    # Reshape to (Num_Bins, Factor) and take mean along axis 1
    s_k_binned = signal_k.reshape(-1, factor).mean(axis=1)
    
    # 2. Calculate Correlation
    # np.corrcoef returns the correlation matrix: [[1, r], [r, 1]]
    # We want the off-diagonal element [0, 1]
    correlation = np.corrcoef(s_k_binned, r_t)[0, 1]
    
    return correlation




