import numpy as np
import matplotlib.pyplot as plt

# Parameters for LIF neuron
V_rest = -74  # Resting membrane potential (mV)
V_thresh = -54  # Threshold potential (mV)
V_reset = -60  # Reset potential after a spike (mV)
tau_m = 20  # Membrane time constant (ms)
E_ex = 0  # Excitatory reversal potential (mV)
tau_ex = 5  # Synaptic conductance decay time (ms)
dt = 0.1  # Time step (ms)
T = 1000  # Total simulation time (ms)

# Parameters for STDP
A_plus = 0.005  # LTP amplitude
A_minus = 0.005 / 1.05  # LTD amplitude
tau_plus = 20  # LTP time constant (ms)
tau_minus = 20  # LTD time constant (ms)

def lif_stdp_simulation():
    """Simulate LIF neuron activity with STDP."""
    time = np.arange(0, T, dt)
    V = np.full_like(time, V_rest, dtype=float)  # Membrane potential
    spikes = np.zeros_like(time, dtype=int)  # Spike times

    # Input synaptic conductance and weights
    g_ex = np.zeros_like(time)
    w = np.random.uniform(0.1, 0.5, size=len(time))  # Initial random weights

    # Generate random input spikes (Poisson process)
    rate = 10  # Firing rate of presynaptic neurons (Hz)
    presynaptic_spikes = np.random.poisson(rate * dt / 1000, size=time.shape)

    # STDP variables
    pre_trace = 0  # Presynaptic spike trace
    post_trace = 0  # Postsynaptic spike trace
    nb_spikes = 0  # Number of spikes 

    for t in range(1, len(time)):
        # Update synaptic conductance (decay + new spikes)
        g_ex[t] = g_ex[t-1] * np.exp(-dt / tau_ex) + presynaptic_spikes[t] * w[t]

        # Compute synaptic current
        I_ex = g_ex[t] * (E_ex - V[t-1])

        # Update membrane potential
        dV = (V_rest - V[t-1] + I_ex) * dt / tau_m
        V[t] = V[t-1] + dV

        # Check for spike
        if V[t] >= V_thresh:
            V[t] = V_reset
            spikes[t] = 1
            post_trace += 1  # Update postsynaptic trace

        # Update STDP traces
        pre_trace *= np.exp(-dt / tau_plus)
        post_trace *= np.exp(-dt / tau_minus)

        if presynaptic_spikes[t]:
            pre_trace += 1  # Update presynaptic trace
            dw = A_plus * np.exp(-post_trace)
            w[t] = min(1.0, w[t] + dw)  # Apply LTP

        if spikes[t]:
            dw = -A_minus * np.exp(-pre_trace)
            w[t] = max(0.0, w[t] + dw)  # Apply LTD
            nb_spikes= nb_spikes+1

    return time, V, spikes, g_ex, w, nb_spikes

# Run the simulation
time, V, spikes, g_ex, w, nb_spikes = lif_stdp_simulation()
print(nb_spikes)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time, V, label='Membrane Potential (V)')
plt.axhline(V_thresh, color='r', linestyle='--', label='Threshold')
plt.axhline(V_rest, color='g', linestyle='--', label='Resting Potential')
plt.ylabel('Membrane Potential (mV)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, g_ex, label='Synaptic Conductance (g_ex)')
plt.ylabel('Conductance')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, spikes, '|', label='Spikes')
plt.ylabel('Spikes')
plt.xlabel('Time (ms)')
plt.legend()

plt.subplot(4, 1, 4)
plt.scatter(time, w, s=10, label='Synaptic Weights')
plt.ylabel('Weights')
plt.xlabel('Time (ms)')
plt.legend()

plt.tight_layout()
plt.show()