#The updated code now simulates a network with feedforward connections only, focusing on:
# Input Layer: Generates spike trains for correlated and uncorrelated groups.
# Feedforward Connections: Random sparse connectivity between input and network neurons.
# STDP: Adjusts the feedforward synaptic weights based on spike timing.
# Plotting : Displays the final feedforward synaptic weights as a grayscale matrix.


import numpy as np
import matplotlib.pyplot as plt

# Parameters for simulation
N_input = 1000  # Number of input neurons
N_network = 250  # Number of network neurons
T = 1000  # Total simulation time (ms)
dt = 0.1  # Time step (ms)

# Parameters for LIF neuron
V_rest = -74  # Resting membrane potential (mV)
V_thresh = -54  # Threshold potential (mV)
V_reset = -60  # Reset potential after a spike (mV)
tau_m = 20  # Membrane time constant (ms)
E_ex = 0  # Excitatory reversal potential (mV)
tau_ex = 5  # Synaptic conductance decay time (ms)

# Parameters for STDP
A_plus = 0.005  # LTP amplitude
A_minus = 0.005 / 1.05  # LTD amplitude
tau_plus = 20  # LTP time constant (ms)
tau_minus = 20  # LTD time constant (ms)

# Connectivity
connection_prob = 0.2  # Probability of connection from input to network neuron

# Initialize synaptic weights
#feedforward_weights = np.random.uniform(0.1, 0.5, size=(N_input, N_network))
#feedforward_mask = np.random.rand(N_input, N_network) < connection_prob
#feedforward_weights *= feedforward_mask  # Apply the connectivity mask


# Initialize synaptic weights
feedforward_weights = np.random.uniform(0.1, 0.5, size=(N_input, N_network))  # Default weights
# Create a mask for neurons 81-120 in the network
selective_neurons = np.arange(80, 120)  # Neurons 81 through 120 (0-indexed)
# Assign higher initial weights for selective neurons
feedforward_weights[:, selective_neurons] = np.random.uniform(0.5, 1.0, size=(N_input, len(selective_neurons)))
# Apply connectivity mask
feedforward_mask = np.random.rand(N_input, N_network) < connection_prob
feedforward_weights *= feedforward_mask  # Apply the connectivity mask


# Generate input spike trains (correlated and uncorrelated)
def generate_input_spikes():
    """Generate spike trains for input neurons."""
    spike_trains = np.zeros((N_input, int(T / dt)))
    
    # Correlated group ((Need to check if the neurons are correlated or not))
    correlated_group = np.arange(500, 1000)  # Neurons 501-1000 are correlated
    uncorrelated_group = np.arange(0, 500)  # Neurons 1-500 are uncorrelated

    # Generate correlated spikes
    base_spike_train = np.random.poisson(10 * dt / 1000, size=int(T / dt))
    for neuron in correlated_group:
        spike_trains[neuron, :] = base_spike_train

    # Generate uncorrelated spikes
    for neuron in uncorrelated_group:
        spike_trains[neuron, :] = np.random.poisson(10 * dt / 1000, size=int(T / dt))

    return spike_trains

# Initialize network state
def initialize_network():
    """Initialize network neuron states."""
    V = np.full(N_network, V_rest, dtype=float)  # Membrane potentials as floats
    spikes = np.zeros(N_network)  # Spike times
    g_ex = np.zeros(N_network)  # Synaptic conductances
    return V, spikes, g_ex

# Simulate feedforward connections with STDP
def simulate_feedforward_only():
    """Simulate network with feedforward connections and STDP."""
    time = np.arange(0, T, dt)
    input_spikes = generate_input_spikes()
    V, spikes, g_ex = initialize_network()

    for t_idx, t in enumerate(time):
        # Update input-driven conductance
        g_ex = g_ex * np.exp(-dt / tau_ex) + np.dot(input_spikes[:, t_idx], feedforward_weights)

        # Update membrane potentials
        dV = (V_rest - V + g_ex * (E_ex - V)) * dt / tau_m
        V += dV

        # Check for spikes
        spiking_neurons = V >= V_thresh
        spikes[spiking_neurons] = 1
        V[spiking_neurons] = V_reset

        # Update feedforward weights with STDP
        for i in range(N_input):
            if input_spikes[i, t_idx]:
                for j in range(N_network):
                    if spikes[j]:
                        feedforward_weights[i, j] += A_plus * np.exp(-1 / tau_plus)
                        feedforward_weights[i, j] = min(feedforward_weights[i, j], 1.0)  # Clamp weights

        for j in range(N_network):
            if spikes[j]:
                for i in range(N_input):
                    if input_spikes[i, t_idx]:
                        feedforward_weights[i, j] -= A_minus * np.exp(-1 / tau_minus)
                        feedforward_weights[i, j] = max(feedforward_weights[i, j], 0.0)  # Clamp weights

    return feedforward_weights

# Run the simulation
final_weights = simulate_feedforward_only()

# Plot the final feedforward weights
plt.figure(figsize=(10, 8))
plt.imshow(final_weights, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label='Synaptic Weight')
plt.xlabel('Network Neurons')
plt.ylabel('Input Neurons')
plt.title('Final Feedforward Synaptic Weights')
plt.show()