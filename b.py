# %% Import all Libraries

import numpy as np
import matplotlib.pyplot as plt

# %% Parameter Initialization:

duration=5000; # duration in ms
dt=0.1; # simulation time step.
tau=50; # Filter time for the input.
tRef=5; # Refractory period for the spike trains.
nn=100; # Number of spiketrains we seek to create later.
spikebin=5; # Number of ms per PSTH bin.

Timevector = np.arange(0.1, duration+dt, dt)
# A vector of time in ms, in steps of dt
WhiteNoise = np.random.rand(len(Timevector))- 0.5
# Uniform white noise drawn from =/- 0.5 of the size of time vector
FilteredWhiteNoise  = np.zeros_like(WhiteNoise)
# An empty vector which will be used to create time-filtered input
SpikeTrains = np.zeros((nn, len(Timevector)))
# Matrix that will hold all spiketrains
avetrain = 0
# A counter to calculate the averaging firing rate
tslt=0
# (== t(ime)s(ince)l(ast)(t)oggle 
# this serves as a Boolean for the sparsification of the input signal.
tsls = np.zeros(nn)
# (== t(ime)s(ince)l(ast)(s)pike
# (to keep track of the refractory period of each spike train)
BinnedSpikeTrains = np.zeros(int(duration / spikebin))
# a vector to create a PSTH with binwidth “spikebin” from the spike trains.


# %% Making the time-filtered white noise signal:
for t in range(1, len(WhiteNoise)):
    FilteredWhiteNoise[t] = (
        WhiteNoise[t] - (WhiteNoise[t] - FilteredWhiteNoise[t - 1]) * np.exp(-dt / tau))

# This routine changes the signal trace ”FilteredWhiteNoise”
# by a ”exp(-dt/tau)” fraction of the difference between the signal
# and a random number.
max_abs = np.max(np.abs(FilteredWhiteNoise))
FilteredWhiteNoise = FilteredWhiteNoise / max_abs
# Normalize to a maximum value of 1.


# %% Plotting:
plt.figure(1)
plt.subplot(4, 1, 1)
plt.plot(Timevector, FilteredWhiteNoise)
plt.axis([0, duration, -1, 1])
x = "Time Filtered White Noise (FWN)"
plt.title(x)
plt.show()

# %% Normalize and Rectify:
FilteredWhiteNoise = FilteredWhiteNoise * (500 * dt / 1000)
# Normalizes the trace to a peak value of 500Hz*dt (=0.05).

FilteredWhiteNoise[FilteredWhiteNoise < 0] = 0
# Sets all negative values of ”FilteredWhiteNoise” to 0.

# %% Plotting:
plt.subplot(4, 1, 2)
plt.plot(Timevector, FilteredWhiteNoise, 'b', linewidth=1.1)

# %% Sparsifieing the Input Signals:
# This routine goes through the signal trace and deletes entire
# ”activity bumps” if certain conditions are fullfilled:

toggle = 0
tslt = 0

for d in range(0,len(TimeVector)-1):
    if FilteredWhiteNoise(d)==0 and toggle==0 and (d-tslt)>10:
        toggle=1
        tslt=d
    elif toggle==1:
        FilteredWhiteNoise(d) = 0

        if FilteredWhiteNoise(d+1)==0 and (d-tslt)>5:
            toggle=0


# %% Plotting:
plt.subplot(4,1,2)
plot(Timevector, Filtered)