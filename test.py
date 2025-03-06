import numpy as np
import matplotlib.pyplot as plt


class STDP_Network:
    def __init__(self, num_neurons=250, num_poisson=250, num_inputs=1000, dt=0.1,
                 tau_pre=20.0, tau_post=20.0, tau_m=20.0, V_rest=-74.0, V_reset=-60.0,
                 V_thresh=-54.0, C_m=0.9, g_leak=0.2, tau_s=5.0,
                 A_plus_ff=0.005, A_minus_ff=0.005, A_plus_recur=0.001, A_minus_recur=0.001,
                 B_ff=1.06, B_recur=1.04, g_max=0.02, poisson_rate=10, stimulus_width=100):
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
        self.tau_s = tau_s  # Synaptic time constant (ms)

        # Feedforward learning rates and scaling factor
        self.A_plus_ff = A_plus_ff
        self.A_minus_ff = A_minus_ff
        self.B_ff = B_ff

        # Recurrent learning rates and scaling factor
        self.A_plus_recur = A_plus_recur
        self.A_minus_recur = A_minus_recur
        self.B_recur = B_recur

        self.g_max = g_max

        self.mean_stim_time = 50
        self.R0 = 10
        self.R1 = 80
        self.sigma = 100

        # Initialize feedforward weights: shape (num_inputs, num_neurons)
        # self.ff_weights = np.random.rand(num_inputs, num_neurons) * g_max
        # self.ff_weights = np.random.uniform(0.4, 1.0, (num_inputs, num_neurons))  # Stronger initial weights
        # Initialize feedforward weights: shape (num_inputs, num_neurons)
        # Define the center point (neuron 700)
        center_neuron = 400

        # Generate Gaussian distribution for weights with peak at neuron 700
        weight_distribution = np.exp(-((np.arange(self.num_inputs) - center_neuron) ** 2) / (2 * stimulus_width ** 2))
        weight_distribution = weight_distribution / np.max(weight_distribution)  # Normalize

        # Initialize weights using this distribution, scaled by the desired maximum weight
        self.ff_weights = np.random.uniform(0.6, 1.0, (self.num_inputs, self.num_neurons)) * weight_distribution[:, None]

        self.ff_weights[:, 100:201] = 0

        '''
        self.ff_weights = np.random.uniform(0.6, 1.0, (self.num_inputs, self.num_neurons)) * weight_distribution[:, None]
        for i in range(num_inputs):
            for j in range(num_neurons):
                d = i / 5 - j
                if d > 100:
                    d = 200 - d
                elif d < -100:
                    d = 200 + d
                self.ff_weights[i, j] = max(0, np.random.uniform(0.6, 1.0 * g_max))
        '''
        # Create and enforce sparsity mask (20% chance) on feedforward connections:
        self.mask = (np.random.rand(num_inputs, num_neurons) < 0.2).astype(int)
        self.ff_weights *= self.mask

        # Initialize recurrent weights (network-to-network); these start at 0.
        self.recur_weights = np.zeros((num_neurons, num_neurons))
        for j in range(num_neurons):
            for k in range(j - 40, j + 41):  # Limited to j-40 to j+40 range
                k_mod = k % num_neurons  # Apply periodic boundary condition
                if k_mod != j:  # Ensure no self-connections
                    self.recur_weights[j, k_mod] = 0  # Initial recurrent weight set to zero

        # Poisson neurons: here we use one per network neuron.
        # Their spikes will directly update the excitatory conductance g_ex.
        self.poisson_rate = poisson_rate
        # g_ex: excitatory conductance for each network neuron.
        self.g_ex = np.zeros(num_neurons)

        # Membrane potential for each network neuron
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
        """
        Each network neuron receives input from one dedicated Poisson neuron.
        For each network neuron, if its Poisson neuron fires (with probability poisson_rate*dt/1000),
        g_ex is updated by a fixed factor (here, 0.096).
        """
        poisson_spikes = np.random.rand(self.num_neurons) < (self.poisson_rate * self.dt / 1000.0)
        self.g_ex = 0.5 * poisson_spikes.astype(float)  # 0.096 in the paper but made it bigger

    def generate_stimulus(self, sim_time):
        """
        Generate an input spike array over simulation time.
        """
        num_steps = int(sim_time / self.dt)
        input_spikes = np.zeros((num_steps, self.num_inputs), dtype=int)
        current_step = 0

        while current_step < num_steps:
            interval_duration = np.random.exponential(scale=self.mean_stim_time)
            interval_steps = max(1, int(interval_duration / self.dt))
            end_step = min(num_steps, current_step + interval_steps)

            s = np.random.uniform(1, 1000)  # Random stimulus, can be set to 700 if needed

            a_vals = np.arange(self.num_inputs)
            rates = self.R0 + self.R1 * (
                    np.exp(-((s - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                    np.exp(-((s + 1000 - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                    np.exp(-((s - 1000 - a_vals) ** 2) / (2 * self.sigma ** 2))
            )

            p = rates * self.dt / 1000.0  # Convert from Hz to spike probability per timestep

            for t in range(current_step, end_step):
                input_spikes[t, :] = (np.random.rand(self.num_inputs) < p).astype(int)

            current_step = end_step

        return input_spikes

    def update_membrane_potential(self, pre_spikes_feed, pre_spikes_recur):
        """
        Update the membrane potential of each network neuron (LIF dynamics).
        Includes:
        - Exponential decay of excitatory conductance g_ex
        - Updates from presynaptic spikes (both feedforward and recurrent)
        - LIF membrane potential update
        """
        # Decay excitatory conductance
        self.g_ex *= np.exp(-self.dt / self.tau_s)

        # Corrected matrix multiplication:
        self.g_ex += np.dot(pre_spikes_feed, self.ff_weights) * 3  # Input spikes to Network neurons
        self.g_ex += np.dot(pre_spikes_recur, self.recur_weights) * 3  # Recurrent spikes to Network neurons

        # Compute synaptic current
        synaptic_current = self.g_ex - self.g_leak * (self.V_mem - self.V_rest)

        # Update membrane potential using LIF dynamics
        dV = (synaptic_current / self.C_m) * (self.dt / self.tau_m)
        self.V_mem += dV

        # Check for spikes
        spike = self.V_mem >= self.V_thresh

        # Reset spiking neurons
        self.V_mem[spike] = self.V_reset
        self.spike_train[spike] = 1  # Mark spike occurrence

    def update_pre_feedforward(self, pre_spikes):
        print("Updating pre feedforward weights...")
        print("Pre-spikes count:", np.sum(pre_spikes))
        self.pre_trace_feed[pre_spikes] += 1
        print("Pre-trace (sample):", self.pre_trace_feed[:10])  # Print first 10 values

        for i in np.where(pre_spikes)[0]:
            self.ff_weights[i, :] -= self.B_ff * self.A_minus_ff * self.post_trace_feed * (self.mask[i, :] == 1)
        self.ff_weights *= self.mask  # Enforce sparsity
        self.ff_weights = np.clip(self.ff_weights, 0, self.g_max)

    def update_post_feedforward(self, post_spikes):
        print("Updating post feedforward weights...")
        print("Post-spikes count:", np.sum(post_spikes))
        self.post_trace_feed[post_spikes] += 1
        print("Post-trace (sample):", self.post_trace_feed[:10])  # Print first 10 values

        for j in np.where(post_spikes)[0]:
            self.ff_weights[:, j] += self.B_ff * self.A_plus_ff * self.pre_trace_feed * (self.mask[:, j] == 1)
        self.ff_weights *= self.mask
        self.ff_weights = np.clip(self.ff_weights, 0, self.g_max)

    def update_recurrent_weights(self, pre_spikes, post_spikes):
        """
        Update recurrent weights based on pre and post spikes.
        """
        # Update pre-synaptic trace for neurons that spiked
        self.pre_trace_recur[pre_spikes, :] += 1

        # Apply STDP depression (A_minus) based on post-trace
        self.recur_weights[pre_spikes, :] -= self.B_recur * self.A_minus_recur * self.post_trace_recur[pre_spikes, :]

        # Update post-trace for every active postsynaptic connection
        self.post_trace_recur[:, post_spikes] += 1

        # Apply STDP potentiation (A_plus) based on pre-trace
        self.recur_weights[:, post_spikes] += self.B_recur * self.A_plus_recur * self.pre_trace_recur[:, post_spikes]

        # Ensure weights remain within valid range
        self.recur_weights = np.clip(self.recur_weights, 0, self.g_max)

        # Clip weights between 0 and g_max
        # self.recur_weights = np.clip(self.recur_weights, 0, self.g_max)

    def simulate(self, T=10, feed_pre_times=[10, 50, 120], feed_post_times=[15, 55, 130],
                 recur_pre_times=[60, 140], recur_post_times=[65, 150]):
        """
        Run the simulation for T time steps.
        """
        weights_history_ff = []  # Track feedforward weights over time
        weights_history_recur = []  # Track recurrent weights over time

        input_spikes = self.generate_stimulus(T)  # Generate input spikes for all time steps

        for t in range(T):
            self.decay_traces()
            self.update_poisson_input()

            self.spike_train.fill(0)  # Reset spike train before each time step

            # Extract a single timestep's spike data
            pre_spikes_feed = input_spikes[t].astype(bool)  # Shape (num_inputs,)
            pre_spikes_recur = self.spike_train.astype(bool)  # Shape (num_neurons,)

            # Update membrane potential with the correct shapes
            self.update_membrane_potential(pre_spikes_feed, pre_spikes_recur)

            # Detect postsynaptic spikes in the current time step
            post_spikes = self.spike_train.astype(bool)

            # Weight updates
            if t in feed_pre_times:
                self.update_pre_feedforward(pre_spikes_feed)
            if t in feed_post_times:
                self.update_post_feedforward(post_spikes)

            self.update_recurrent_weights(pre_spikes_recur, post_spikes)

            # Store weight history
            weights_history_ff.append(self.ff_weights.copy())
            weights_history_recur.append(self.recur_weights.copy())

        return np.array(weights_history_ff), np.array(weights_history_recur)

# Run the simulation
network = STDP_Network()
ff_history, recur_history = network.simulate()

# Plotting final feedforward and recurrent weight distributions
plt.figure(figsize=(8, 6))
plt.imshow(ff_history[-1], aspect='auto', cmap='gray_r',
           extent=[0, network.num_neurons, 0, network.num_inputs])
plt.colorbar(label="Feedforward Weight Strength")
plt.xlabel("Network Neuron")
plt.ylabel("Input Neuron")
plt.title("Final Feedforward Synaptic Weight Distribution")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(recur_history[-1], aspect='auto', cmap='gray',
           extent=[0, network.num_neurons, 0, network.num_neurons])
plt.colorbar(label="Recurrent Weight Strength")
plt.xlabel("Network Neuron (Postsynaptic)")
plt.ylabel("Network Neuron (Presynaptic)")
plt.title("Final Recurrent Synaptic Weight Distribution")
plt.show()