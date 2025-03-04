import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_input_neurons = 1000
num_network_neurons = 250
g_max = 0.02
A_plus = 0.001
A_minus = A_plus * 1.05
tau_plus = 20
tau_minus = 20
simulation_time = 1000  # ms
dt = 1  # ms
time_window = 20  # ms

# LIF Parameters
C_m = 20  # Membrane capacitance (ms)
V_rest = -74  # Resting potential (mV)
V_thresh = -50  # Spiking threshold (mV)
V_reset = -60  # Reset potential (mV)
E_ex = 0  # Excitatory reversal potential (mV)
tau_ex = 5  # Synaptic decay constant (ms)
b_rate = 500  # Background Poisson input (Hz)

# Rate function parameters for input neurons
R0 = 10
R1 = 80
sigma = 100
s = 700  # Stimulus location of the input neuron

# Generate firing rates for input neurons based on the given rate function
def generate_input_firing_rates():
    firing_rates = np.zeros(num_input_neurons)
    for i in range(num_input_neurons):
        rate = R0 + R1 * (np.exp(-((s - i) ** 2) / (2 * sigma ** 2)) +
                          np.exp(-((s + 1000 - i) ** 2) / (2 * sigma ** 2)) +
                          np.exp(-((s - 1000 - i) ** 2) / (2 * sigma ** 2)))
        firing_rates[i] = rate
    return firing_rates

# Generate spikes for input neurons using Poisson process
def generate_input_spikes(firing_rates, time_steps):
    spikes = np.zeros((num_input_neurons, time_steps))
    for i in range(num_input_neurons):
        spike_prob = firing_rates[i] * dt / 1000  # Convert firing rate to spike probability
        spikes[i, :] = np.random.rand(time_steps) < spike_prob
    return spikes

# Define the STDP window function
def stdp_window(delta_t):
    return np.where(delta_t < 0,
                    A_plus * np.exp(delta_t / tau_plus),
                    -A_minus * np.exp(-delta_t / tau_minus))

# LIF Model for network neurons
def lif_model(spikes_input, num_network_neurons, simulation_time, dt, g_max, synaptic_decay, b_rate):
    # Initial conditions
    V = np.full(num_network_neurons, V_rest, dtype=np.float64)  # Membrane potential (ensure it's float64)
    spikes_network = np.zeros((num_network_neurons, simulation_time), dtype=np.int32)  # Track network neuron spikes
    synaptic_weights = np.zeros((num_input_neurons, num_network_neurons), dtype=np.float64)  # Synaptic weights
    spike_times = {j: [] for j in range(num_network_neurons)}  # Record spike times

    # Background Poisson spike
    b_spikes = np.random.rand(num_network_neurons, simulation_time) < b_rate * dt / 1000

    # Update rule for the network neurons
    for t in range(simulation_time):
        # Excitatory synaptic input
        I_syn = np.dot(synaptic_weights.T, spikes_input[:, t])  # Input from the presynaptic neurons

        # Background Poisson input
        I_syn += np.dot(b_spikes, np.ones(num_input_neurons)) * 0.116

        # Membrane potential update (LIF equation)
        dV = (-(V - V_rest) + I_syn * C_m) * dt / C_m
        V += dV  # Update membrane potential

        # Check for spikes
        spike_occurred = V >= V_thresh
        spikes_network[:, t] = spike_occurred.astype(int)  # Store the spikes as 0 or 1

        # Reset membrane potential after spike
        V[spike_occurred] = V_reset

        # Record spike times
        for neuron in range(num_network_neurons):
            if spike_occurred[neuron]:
                spike_times[neuron].append(t)

        # Update synaptic weights using STDP rule
        for i in range(num_input_neurons):
            for j in range(num_network_neurons):
                if spikes_input[i, t] > 0:  # If the input neuron spikes at this time step
                    for pre_spike_time in spike_times[j]:  # If the network neuron spikes
                        delta_t = t - pre_spike_time
                        weight_update = stdp_window(delta_t)
                        synaptic_weights[i, j] += weight_update
                        synaptic_weights[i, j] = np.clip(synaptic_weights[i, j], 0, g_max)  # Clipping to the max weight

    return synaptic_weights


# Generate input firing rates
firing_rates_input = generate_input_firing_rates()

# Generate input spikes for the entire simulation time
input_spikes = generate_input_spikes(firing_rates_input, simulation_time)

# Precompute synaptic decay factor
synaptic_decay = np.exp(-dt / tau_ex)

# Run the LIF model with STDP
synaptic_weights = lif_model(input_spikes, num_network_neurons, simulation_time, dt, g_max, synaptic_decay, b_rate)

# Normalize and plot the synaptic weights
plt.figure(figsize=(10, 6))

# Normalize the synaptic weights for better visualization
feedforward_strengths_normalized = synaptic_weights / g_max

# Get the coordinates and strengths of non-zero synapses
y, x = np.where(feedforward_strengths_normalized > 0)  # y: input neurons, x: network neurons
weights = feedforward_strengths_normalized[y, x]  # Synaptic strengths

# Plot the scatter points
plt.scatter(x, y, c=weights, cmap='gray_r', s=10, marker='.', vmin=0, vmax=1)
plt.xlabel('Network Neuron')
plt.ylabel('Input Neuron')
plt.title('Feedforward Synaptic Strengths After STDP')

cbar = plt.colorbar(label='g/g_max')
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.xlim(0, 250)  # Network neurons
plt.ylim(0, 1000)  # Input neurons

plt.show()