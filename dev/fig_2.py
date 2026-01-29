import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==================== 1. Configuration & Signal Generation ====================
n_trials = 100
T_sim = 10000.0  # 10 seconds
dt = 5.0         # Bin size for PSTH (ms)
n_groups = 8
N_exc = 100      # Input neurons per group
resolution = 0.1

# Generate Frozen Noise (The "Input Signal" k)
# We generate this ONCE and reuse it for all 100 trials
np.random.seed(42)
time_axis = np.arange(0, T_sim, resolution)
steps = len(time_axis)

input_signals = []
# Create 8 channels with different modulations
for g in range(n_groups):
    # Ornstein-Uhlenbeck / Filtered Noise style signal
    noise = np.random.normal(0, 1, steps)
    s = np.zeros(steps)
    tau_s = 50.0 # ms
    alpha = np.exp(-resolution / tau_s)
    
    for t in range(1, steps):
        s[t] = s[t-1] * alpha + noise[t] * (1-alpha)
    
    # Rectify and Scale
    s = s - np.min(s)
    s = s / np.max(s) * 30.0 # Max rate fluctuation 30Hz
    s += 5.0 # Base rate 5Hz
    
    # Make channel 4 the "Preferred" one (Higher amplitude/mean)
    if g == 4:
        s *= 1.5 
        
    input_signals.append(s)

# ==================== 2. Simulation Loop (100 Trials) ====================

all_spike_times = []
trial_indices = []

print(f"Starting {n_trials} trials of {T_sim/1000}s each...")

# We need the weights to calculate conductance later
w_exc_profile = [] 

for trial in range(n_trials):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution, "local_num_threads": 1, "print_time": False})
    
    # --- Network Construction ---
    neuron_params = {
        "C_m": 200.0, "g_L": 10.0, "E_L": -60.0, 
        "V_th": -50.0, "V_reset": -60.0, "t_ref": 5.0,
        "E_ex": 0.0, "E_in": -80.0
    }
    
    post = nest.Create("iaf_cond_alpha", 1, params=neuron_params)
    spikedetector = nest.Create("spike_recorder")
    nest.Connect(post, spikedetector)
    
    # Input Channels
    w_exc_profile = [] # Store for plotting later
    
    for g in range(n_groups):
        # Create Poisson Gen using the FROZEN signal
        gen = nest.Create("inhomogeneous_poisson_generator")
        nest.SetStatus(gen, {
            "rate_times": time_axis + resolution,
            "rate_values": input_signals[g],
            "allow_offgrid_times": True
        })
        
        # Parrots (Population)
        parrots = nest.Create("parrot_neuron", N_exc)
        nest.Connect(gen, parrots, "all_to_all")
        
        # Gaussian Tuning for weights
        sigma = 1.5
        dist = min(abs(g - 4), n_groups - abs(g - 4))
        w_val = 1.0 + 3.0 * np.exp(-(dist**2)/(2*sigma**2))
        w_exc_profile.append(w_val)
        
        # Connect to Post (Fixed Weights for this demo)
        nest.Connect(parrots, post, "all_to_all", syn_spec={"weight": w_val, "delay": 1.0})
        
        # Add static inhibition to balance slightly
        nest.Connect(parrots, post, "all_to_all", syn_spec={"weight": -0.5, "delay": 1.0})

    # --- Run ---
    nest.Simulate(T_sim)
    
    # --- Collect Data ---
    events = nest.GetStatus(spikedetector, "events")[0]
    spikes = events["times"]
    
    all_spike_times.extend(spikes)
    trial_indices.extend([trial] * len(spikes))
    
    if (trial+1) % 10 == 0:
        print(f"Completed trial {trial+1}/{n_trials}")

# ==================== 3. Analysis: PSTH & Correlation ====================

# A. Calculate PSTH (Instantaneous Firing Rate)
bins = np.arange(0, T_sim + dt, dt)
counts, _ = np.histogram(all_spike_times, bins=bins)
# Rate (Hz) = count / (bin_width_sec * n_trials)
psth_rate = counts / ((dt/1000.0) * n_trials)

# B. Calculate Correlation per Channel
correlations = []

# We need to downsample input signals to match PSTH bin size (dt = 1ms)
# Original resolution = 0.1ms. Factor = 10.
downsample_factor = int(dt / resolution)

for g in range(n_groups):
    # Downsample input signal k
    sig_k = input_signals[g]
    # Simple averaging for downsampling
    sig_k_binned = np.mean(sig_k.reshape(-1, downsample_factor), axis=1)
    
    # Trim to match sizes if necessary (due to rounding)
    min_len = min(len(psth_rate), len(sig_k_binned))
    
    # Calculate Pearson Correlation
    # corr matrix returns [[1, r], [r, 1]]
    r = np.corrcoef(psth_rate[:min_len], sig_k_binned[:min_len])[0, 1]
    correlations.append(r)

# ==================== 4. Plotting ====================

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

# --- Plot 1: Raster Plot (Trial vs Time) ---
ax0 = plt.subplot(gs[0])
ax0.scatter(all_spike_times, trial_indices, s=1, c='k', alpha=0.5)
ax0.set_xlabel("Time (ms)")
ax0.set_ylabel("Trial #")
ax0.set_title("Raster Plot (100 Repeats with Frozen Input)")
ax0.set_xlim(0, 1000) # Zoom in on first second for visibility
ax0.set_ylim(-1, n_trials)

# --- Plot 2: Correlation & Conductance ---
ax1 = plt.subplot(gs[1])

# Create double axis: Left=Conductance, Right=Correlation
ax2 = ax1.twinx()

channels = np.arange(1, n_groups + 1)

# Theoretical Conductance (Weight * Mean Rate)
# Just approximating relative conductance strength based on weights here
cond_values = np.array(w_exc_profile) * 5.0 # Scaling for visualization

# Plot Conductance (Bar)
bars = ax1.bar(channels, cond_values, color='lightgray', label='Conductance (Weight)', alpha=0.6)

# Plot Correlation (Line/Points)
lines = ax2.plot(channels, correlations, 'ro-', linewidth=2, label='Correlation w/ Signal')

ax1.set_xlabel("Input Channel")
ax1.set_ylabel("Exc. Conductance / Weight (nS)", color='gray')
ax2.set_ylabel("Correlation Coeff (r)", color='red')
ax2.set_ylim(-0.1, 1.0)
ax1.set_title("Channel Properties")

# Legend
lns = [bars, lines[0]]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

plt.tight_layout()
plt.show()

print("Correlation Values per Channel:", np.round(correlations, 3))