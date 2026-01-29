
"""Functions to create neurons and weights in NEST."""

import nest
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
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
    
    return rate_measured


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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_input_raster_and_weights(signals, weights_list, dt_sim=0.1, bin_size_ms=5.0):
    """
    Plota o raster (heatmap) dos sinais de entrada e os pesos correspondentes.
    
    Args:
        signals (np.array): Array 3D [n_canais, n_trials, n_steps]
        weights_list (list/array): Lista de pesos excitatórios por canal.
        dt_sim (float): Passo de tempo da simulação original (ex: 0.1 ms).
        bin_size_ms (float): Tamanho do bin desejado para o plot (ex: 5.0 ms).
    """
    
    # 1. Preparação dos Dados (Binagem)
    # Selecionamos a Trial 0 (pois o sinal de entrada s_k(t) é o mesmo padrão repetido)
    # Shape alvo: [n_canais, n_steps]
    raw_signal = signals[:, 0, :] 
    
    n_channels, n_steps_total = raw_signal.shape
    
    # Calcular fator de redução (ex: 5ms / 0.1ms = 50 steps)
    factor = int(bin_size_ms / dt_sim)
    
    # Arredondar o tamanho do array para ser divisível pelo fator
    limit = (n_steps_total // factor) * factor
    raw_signal_trimmed = raw_signal[:, :limit]
    
    # Reshape e média para fazer o downsampling (Binagem)
    # Transforma [Canais, Tempo] -> [Canais, Novos_Bins, Fator] -> Média no eixo do Fator
    signal_binned = raw_signal_trimmed.reshape(n_channels, -1, factor).mean(axis=2)
    
    # Eixo de tempo para o plot
    n_bins = signal_binned.shape[1]
    t_max = n_bins * bin_size_ms
    
    # 2. Configuração do Plot (GridSpec)
    fig = plt.figure(figsize=(14, 6))
    # width_ratios=[4, 1] faz o raster ser 4x mais largo que o gráfico de pesos
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    # --- PAINEL ESQUERDO: Raster (Heatmap) ---
    ax_raster = plt.subplot(gs[0])
    
    # imshow plota a matriz como cores
    # aspect='auto' é vital para gráficos de tempo longo
    # origin='upper' coloca o canal 0 no topo
    im = ax_raster.imshow(signal_binned, 
                          aspect='auto', 
                          cmap='inferno', # 'inferno', 'hot', ou 'viridis' funcionam bem
                          interpolation='nearest',
                          extent=[0, t_max, n_channels - 0.5, -0.5]) # Ajuste fino para alinhar ticks
    
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Input Channel #")
    ax_raster.set_title(f"Input Firing Rates (Bin = {bin_size_ms}ms)")
    
    # Ajustar Ticks do Y para mostrar número dos canais inteiros
    ax_raster.set_yticks(np.arange(n_channels))
    ax_raster.invert_yaxis() # Garante que Canal 0 fique no topo (opcional, padrão de matriz)
    
    # Barra de cores
    cbar = plt.colorbar(im, ax=ax_raster, pad=0.02)
    cbar.set_label("Rate (Hz)")
    
    # --- PAINEL DIREITO: Pesos Excitatórios ---
    ax_weights = plt.subplot(gs[1], sharey=ax_raster)
    
    y_pos = np.arange(n_channels)
    
    # Plota barras horizontais
    # A cor 'k' (preto) com alpha ajuda a visualizar
    ax_weights.barh(y_pos, weights_list, color='black', alpha=0.7, height=0.6)
    
    ax_weights.set_xlabel("Excitatory Weight (nS)")
    ax_weights.set_title("Weights Profile")
    ax_weights.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Esconde os labels Y do gráfico da direita (pois já estão na esquerda)
    plt.setp(ax_weights.get_yticklabels(), visible=False)
    
    plt.tight_layout()
    plt.show()

