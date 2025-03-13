import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


class STDP_Network:
    def __init__(self, num_neurons=10, num_poisson=250, num_inputs=1000, dt=0.1,
                 tau_pre=20.0, tau_post=20.0, tau_m=20.0, V_rest=-74.0, V_reset=-60.0,
                 V_thresh=-54.0, C_m=0.9, g_leak=0.2, tau_s=5.0,
                 A_plus_ff=0.005, A_minus_ff=0.005, A_plus_recur=0.001, A_minus_recur=0.001,
                 B_ff=1.06, B_recur=1.04, g_max=0.03, poisson_rate=500, stimulus_width=100,
                 mean_stim_time=100, R0=10, R1=80, sigma=100, sim_time=100):
        """
        Initialize the network with LIF neuron dynamics for the network neurons (not input neurons).
        """
        self.num_neurons = num_neurons  # Postsynaptic (network) neurons
        self.num_inputs = num_inputs  # Presynaptic (input) neurons
        self.dt = dt
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # LIF neuron parameters
        self.tau_m = tau_m  # Membrane time constant (ms)
        self.V_rest = V_rest  # Resting potential (mV)
        self.V_reset = V_reset  # Reset potential (mV)
        self.V_thresh = V_thresh  # Threshold potential (mV)
        self.C_m = C_m  # Membrane capacitance
        self.g_leak = g_leak  # Leak conductance
        self.E_ex = 0
        self.tau_s = tau_s  # Synaptic time constant (ms)
        self.mean_stim_time = mean_stim_time
        self.R0 = R0
        self.R1 = R1
        self.sigma = sigma
        # Feedforward learning rates and scaling factor
        self.A_plus_ff = A_plus_ff
        self.A_minus_ff = A_minus_ff
        self.B_ff = B_ff
        self.sim_time = sim_time

        # Recurrent learning rates and scaling factor
        self.A_plus_recur = A_plus_recur
        self.A_minus_recur = A_minus_recur
        self.B_recur = B_recur

        self.g_max = g_max

        self.ff_weights = np.random.uniform(0, self.g_max,
                                            size=(self.num_inputs, self.num_neurons))  # * weight_distribution[:, None]
        #self.ff_weights[:, 100:201] = 0

        # Create and enforce sparsity mask (20% chance) on feedforward connections:
        self.mask = (np.random.rand(num_inputs, num_neurons) < 0.2).astype(int)
        #self.mask[:, 100:201] = 0
        self.ff_weights *= self.mask

        # Initialize recurrent weight matrix (network-to-network); these start at 0.
        self.recur_weights = np.zeros((num_neurons, num_neurons))

        # Poisson neurons: here we use one per network neuron.
        self.poisson_rate = poisson_rate
        self.g_ex = np.zeros(num_neurons)

        # Membranpotentiale & Spiketrain initialise / Membrane potential for each network neuron
        self.V_mem = np.full(num_neurons, V_rest)  # Initialize to resting potential
        self.spike_train = np.zeros(num_neurons)  # Track spikes

        # Stimulus properties (for generating input spike patterns)
        self.stimulus_width = stimulus_width
        self.preferred_stimulus = np.linspace(0, 1000, num_inputs)
        self.current_stimulus = None

        # Traces for feedforward updates per connection:
        self.pre_trace_feed = np.zeros((self.num_inputs, self.num_neurons))
        self.post_trace_feed = np.zeros((self.num_inputs, self.num_neurons))

        # Traces for recurrent updates per connection:
        self.pre_trace_recur = np.zeros((self.num_neurons, self.num_neurons))
        self.post_trace_recur = np.zeros((self.num_neurons, self.num_neurons))

    def decay_traces(self):
        """Decay all traces over one time step."""
        self.pre_trace_feed *= np.exp(-self.dt / self.tau_pre)
        self.post_trace_feed *= np.exp(-self.dt / self.tau_post)
        self.pre_trace_recur *= np.exp(-self.dt / self.tau_pre)
        self.post_trace_recur *= np.exp(-self.dt / self.tau_post)

    def update_poisson_input(self):
        """Each network neuron receives input from one dedicated Poisson neuron"""
        poisson_spikes = np.random.rand(self.num_neurons) < (self.poisson_rate * (self.dt / 1000.0))
        # self.g_ex += 0.116 * poisson_spikes.astype(int)

    def generate_stimulus(self):
        interval_duration = np.random.exponential(scale=self.mean_stim_time)
        interval_steps = max(1, int(interval_duration / self.dt))
        spike_array = np.zeros((interval_steps, self.num_inputs), dtype=int)

        s = np.random.uniform(1, 1000)

        a_vals = np.arange(self.num_inputs)
        rates = self.R0 + self.R1 * (
                np.exp(-((s - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                np.exp(-((s + 1000 - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                np.exp(-((s - 1000 - a_vals) ** 2) / (2 * self.sigma ** 2))
        )

        p = rates * self.dt / 1000.0
        for t in range(0, interval_steps):
            spike_array[t, :] = (np.random.rand(self.num_inputs) < p).astype(int)

        return spike_array.tolist()

    def ramp_up_stimulus(self, duration=50):
        spike_array = []
        for s in range(1000):
            a_vals = np.arange(self.num_inputs)
            rates = self.R0 + self.R1 * (
                    np.exp(-((s - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                    np.exp(-((s + 1000 - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                    np.exp(-((s - 1000 - a_vals) ** 2) / (2 * self.sigma ** 2))
            )

            p = rates * self.dt / 1000.0

            for t in range(duration):
                spike_array.append((np.random.rand(self.num_inputs) < p).astype(int))

        return np.array(spike_array)

    def update_membrane_potential(self, pre_spikes_feed, pre_spikes_recur):
        """Update the membrane potential of each network neuron (LIF dynamics)."""
        self.g_ex *= np.exp(-self.dt / self.tau_s)
        self.g_ex += np.dot(pre_spikes_feed, self.ff_weights)  # Input spikes to Network neurons
        self.g_ex += np.dot(pre_spikes_recur, self.recur_weights)  # Recurrent spikes to Network neurons

        # Update membrane potential using LIF dynamics
        dV = (1 / self.tau_m) * (self.V_rest - self.V_mem * (1 + self.g_ex))
        self.V_mem += dV

        # Check for spikes
        spike = self.V_mem >= self.V_thresh

        # Reset spiking neurons
        self.V_mem[spike] = self.V_reset
        self.spike_train[spike] = 1  # Mark spike occurrence

    def update_feedforward_weights(self, pre_spikes, post_spikes):
        """Update feedforward synaptic weights based on STDP rules"""
        # Update pre-synaptic trace for neurons that spiked
        pre_depression = np.zeros_like(self.ff_weights)
        post_potentiation = np.zeros_like(self.ff_weights)

        self.pre_trace_feed[pre_spikes] += 1

        # Apply STDP depression for pre-synaptic spikes
        for i in np.where(pre_spikes)[0]:  # Loop through presynaptic neurons that spiked
            # stdp_factor = np.exp(-self.post_trace_feed / self.tau)  # Exponentiell factor
            delta = self.g_max * self.B_ff * self.A_minus_ff * self.post_trace_feed[i, :]
            pre_depression[i, :] = delta
            self.ff_weights[i, :] -= delta

        # Update post-synaptic trace for neurons that spiked
        self.post_trace_feed[:, post_spikes] += 1

        # Apply STDP potentiation for post-synaptic spikes
        for j in np.where(post_spikes)[0]:  # Iterate over postsynaptic neurons that spiked
            # stdp_factor = np.exp(-self.pre_trace_feed / self.tau)
            delta = self.g_max * self.B_ff * self.A_plus_ff * self.pre_trace_feed[:, j]  
            post_potentiation[:, j] = delta
            self.ff_weights[:, j] += delta

        self.ff_weights *= self.mask

        # Enforce sparsity mask and clip values to [0, g_max]
        self.ff_weights = np.clip(self.ff_weights, 0, self.g_max)

        return pre_depression, post_potentiation

    def update_recurrent_weights(self, pre_spikes, post_spikes):
        """Update recurrent weights based on pre and post spikes."""
        # Initialize matrices to record changes
        pre_depression = np.zeros_like(self.recur_weights)
        post_potentiation = np.zeros_like(self.recur_weights)

        # Update pre-synaptic trace for neurons that spiked
        self.pre_trace_recur[pre_spikes] += 1

        # STDP Depression: For each presynaptic spike, subtract based on the corresponding post-trace.
        for i in np.where(pre_spikes)[0]:
            # Use only the i-th row of post_trace_recur
            delta = self.g_max * self.B_recur * self.A_minus_recur * self.post_trace_recur[i, :]
            pre_depression[i, :] = delta
            self.recur_weights[i, :] -= delta

        # Update post-synaptic trace for neurons that spiked
        self.post_trace_recur[post_spikes] += 1

        # STDP Potentiation: For each postsynaptic spike, add based on the corresponding pre-trace.
        for j in np.where(post_spikes)[0]:
            # Use only the j-th column of pre_trace_recur
            delta = self.g_max * self.B_recur * self.A_plus_recur * self.pre_trace_recur[:, j]
            post_potentiation[:, j] = delta
            self.recur_weights[:, j] += delta

        # Clip the recurrent weights to the allowed range.
        self.recur_weights = np.clip(self.recur_weights, 0, self.g_max)

        return pre_depression, post_potentiation

    def simulate(self, T=1000000, ramp_up_time=50):
        """
        Run the simulation for T time steps.
        """
        network_spike_history = []  # Record network spikes over time

        # Precompute ramp-up stimulus
        ramp_stimulus = self.ramp_up_stimulus(duration=ramp_up_time)
        input_stimulus = []

        for t in tqdm(range(T)):
            if len(input_stimulus) == 0:
                input_stimulus = self.generate_stimulus()

            self.decay_traces()
            #self.update_poisson_input()
            input_spikes = input_stimulus.pop()

            self.spike_train.fill(0)  # Reset spike train before each time step

            pre_spikes_feed = np.array(input_spikes, dtype=bool)
            pre_spikes_recur = self.spike_train.astype(bool)
            # Update membrane potentials with proper conductance dynamics
            self.update_membrane_potential(pre_spikes_feed, pre_spikes_recur)

            #network_spike_history.append(self.spike_train.copy())

            post_spikes = self.spike_train.astype(bool)
            post_spikes_recurrent = self.spike_train.astype(bool)

            # Update feedforward weights
            self.update_feedforward_weights(pre_spikes_feed, post_spikes)
    
            # Update recurrent weights and capture the contributions
            pre_dep, post_pot = self.update_recurrent_weights(post_spikes, post_spikes_recurrent)

        for t in tqdm(range(ramp_up_time*self.num_inputs)):
            network_spike_history.append(self.spike_train.copy())
            input_spikes = ramp_stimulus[t]
            self.spike_train.fill(0)
            pre_spikes_feed = np.array(input_spikes, dtype=bool)
            pre_spikes_recur = self.spike_train.astype(bool)
            # all_spikes.append(self.spike_train.copy())
            self.update_membrane_potential(pre_spikes_feed, pre_spikes_recur)

        # return (all_spikes)

        return (np.array(network_spike_history))


# Run the simulation
network = STDP_Network()
net_spike_history = network.simulate()
# net_spike_history = network.simulate()
'''
test_stimulus = np.empty((0, 1000))
for i in range(10):
    print(np.array(network.generate_stimulus()).shape)
    test_stimulus = np.concatenate((test_stimulus, network.generate_stimulus()), axis = 0)
'''

# spike_history = np.array(test_stimulus)
spike_history = np.array(network.generate_stimulus())
spike_times_list = [np.where(spike_history[:, neuron] == 1)[0] for neuron in range(spike_history.shape[1])]
#stimulus_spike_train = network.ramp_up_stimulus(duration=100)
network_spike_times_list = [np.where(net_spike_history[:, neuron] == 1)[0] for neuron in
                            range(net_spike_history.shape[1])]

# Convert spike history to firing rates
firing_rates = net_spike_history.mean(axis=1)  # Mean spike count per time step

# Apply Gaussian smoothing
sigma = 20  # Smoothing window in time steps
smoothed_firing_rates = gaussian_filter1d(firing_rates, sigma=sigma, axis=0)

# Plot the firing rate over time
plt.figure(figsize=(10, 5))
plt.plot(smoothed_firing_rates, color="blue")
plt.xlabel("Time (ms)")
plt.ylabel("Firing Rate (spikes/ms)")
plt.title("Network Neuron Response Over Time")
plt.show()

plt.figure(figsize=(10, 6))


# Define time window size (each stimulus lasts 50 time steps)
window_size = 50
num_stimuli = 1000  #number of input neurons

# Compute tuning curve by summing spikes in each window
tuning_curves = []
for k in range(num_stimuli):
    window = net_spike_history[k * window_size:(k + 1) * window_size]  # Get time window
    tuning = window.sum(axis=0)  # Sum spikes over time for each neuron
    tuning_curves.append(tuning)

tuning_curves = np.array(tuning_curves)  # Shape: (1000 stimuli, num_neurons)

# this could be used for periodic boundary conditions (circular shift)
#tuning_curves = np.roll(tuning_curves, shift=num_stimuli // 2, axis=0)


# Apply Gaussian smoothing
sigma = 20
smoothed_curves = gaussian_filter1d(tuning_curves, sigma=sigma, axis=0)

# Plot tuning curves
plt.figure(figsize=(8, 6))
for k in range(network.num_neurons):
    if smoothed_curves[:, k].max() > 0:
        plt.plot(smoothed_curves[:, k] / smoothed_curves[:, k].max(), label=f'Neuron {k}', linewidth=2)

plt.xlabel("Stimulus Number")
plt.ylabel("Normalized Response")
plt.title("Tuning Curves of Network Neuron(s)")
plt.show()

# Compute preferred stimulus for each neuron
preferred_locations = [smoothed_curves[:, k].argmax() for k in range(network.num_neurons)]

# Scatter plot of preferred stimulus locations
plt.figure(figsize=(8, 4))
plt.scatter(range(len(preferred_locations)), preferred_locations, color="blue", alpha=0.7)
plt.xlabel("Neuron Index")
plt.ylabel("Preferred Stimulus")
plt.title("Preferred Stimuli of Network Neurons")
plt.show()


# input spikes
plt.figure(figsize=(10, 6))
plt.eventplot(spike_times_list, colors="black")
plt.xlabel("Time (ms)")
plt.ylabel("Input Neuron")
plt.title("Spike Train Event Plot of Input neurons")
plt.show()
