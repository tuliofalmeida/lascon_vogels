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
