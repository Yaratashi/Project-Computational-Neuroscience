import numpy as np
import matplotlib.pyplot as plt


class STDP_Network:
    def __init__(self, num_neurons=250, num_poisson=250, num_inputs=1000, dt=1.0,
                 tau_pre=20.0, tau_post=20.0, tau_m=20.0, V_rest=-74.0, V_reset=-60.0,
                 V_thresh=-54.0, C_m=1.0, g_leak=0.5, tau_s=5.0,
                 A_plus_ff=0.005, A_minus_ff=0.005, A_plus_recur=0.001, A_minus_recur=0.001,
                 B_ff=1.06, B_recur=1.04, g_max=1.1, poisson_rate=10, stimulus_width=100):
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

        # Initialize feedforward weights based on distance rule
        self.ff_weights = np.zeros((num_inputs, num_neurons))
        for i in range(self.num_inputs):
            for j in range(self.num_neurons):
                d = i / 5 - j
                if d > 100:
                    d = 200 - d
                elif d < -100:
                    d = 200 + d
                self.ff_weights[i, j] = (d / 100) * (0.5 * self.g_max)

        # Enforce sparsity (20% chance of having a connection)
        self.mask = (np.random.rand(num_inputs, num_neurons) < 0.2).astype(int)
        self.ff_weights *= self.mask

        # Initialize recurrent weights: limited to ±40 range
        self.recur_weights = np.zeros((num_neurons, num_neurons))
        for j in range(self.num_neurons):
            for i in range(j - 40, j + 40):
                if 0 <= i < self.num_neurons:
                    self.recur_weights[i, j] = np.random.uniform(0, 0.5 * self.g_max)

        # Add inhibitory connections: -0.3 * g_max
        inhibitory_strength = 0.3 * self.g_max
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    self.recur_weights[i, j] -= inhibitory_strength  # Negative for inhibition

        # Poisson neurons
        self.poisson_rate = poisson_rate
        self.g_ex = np.zeros(num_neurons)  # Excitatory conductance

        # Membrane potential
        self.V_mem = np.full(num_neurons, V_rest)
        self.spike_train = np.zeros(num_neurons)

        # Stimulus properties
        self.stimulus_width = stimulus_width
        self.preferred_stimulus = np.linspace(0, 1000, num_inputs)
        self.current_stimulus = None

        # Traces for learning
        self.pre_trace_feed = np.zeros(num_inputs)
        self.post_trace_feed = np.zeros(num_neurons)
        self.pre_trace_recur = np.zeros(num_neurons)
        self.post_trace_recur = np.zeros(num_neurons)

    def update_poisson_input(self):
        poisson_spikes = np.random.rand(self.num_neurons) < (self.poisson_rate * self.dt / 1000.0)
        self.g_ex = 0.5 * poisson_spikes.astype(float)  # Increased conductance update

    def generate_stimulus(self):
        self.current_stimulus = 700
        rates = 10 + 80 * np.exp(-((self.preferred_stimulus - self.current_stimulus) ** 2) /
                                 (2 * self.stimulus_width ** 2))
        return np.random.poisson(rates / 1000.0)

    def update_membrane_potential(self, pre_spikes_feed, pre_spikes_recur):
        self.g_ex *= np.exp(-self.dt / self.tau_s)
        self.g_ex += np.dot(pre_spikes_feed, self.ff_weights)
        self.g_ex += np.dot(pre_spikes_recur, self.recur_weights)

        synaptic_current = self.g_ex - self.g_leak * (self.V_mem - self.V_rest)
        dV = (synaptic_current / self.C_m) * (self.dt / self.tau_m)
        self.V_mem += dV

        spike = self.V_mem >= self.V_thresh
        self.V_mem[spike] = self.V_reset
        self.spike_train[spike] = 1

    def update_recurrent_weights(self, pre_spikes, post_spikes):
        pre_spikes = pre_spikes.astype(bool)
        post_spikes = post_spikes.astype(bool)

        self.pre_trace_recur[pre_spikes] += 1
        for i in np.where(pre_spikes)[0]:
            self.recur_weights[i, :] -= self.B_recur * self.A_minus_recur * self.post_trace_recur[np.newaxis, :]

        self.post_trace_recur[post_spikes] += 1
        for j in np.where(post_spikes)[0]:
            self.recur_weights[:, j] += self.B_recur * self.A_plus_recur * self.pre_trace_recur[:, np.newaxis]

        self.recur_weights = np.clip(self.recur_weights, 0, self.g_max)

    def simulate(self, T=1500):
        weights_history_ff = []
        weights_history_recur = []

        for t in range(T):
            self.update_poisson_input()
            input_spikes = self.generate_stimulus()

            self.spike_train.fill(0)
            pre_spikes_feed = input_spikes.astype(bool)
            pre_spikes_recur = self.spike_train.astype(bool)

            self.update_membrane_potential(pre_spikes_feed, pre_spikes_recur)
            post_spikes = self.spike_train.astype(bool)

            self.update_recurrent_weights(pre_spikes_recur, post_spikes)

            weights_history_ff.append(self.ff_weights.copy())
            weights_history_recur.append(self.recur_weights.copy())

        return np.array(weights_history_ff), np.array(weights_history_recur)


# Run the simulation
network = STDP_Network()
ff_history, recur_history = network.simulate()

# Plot feedforward weights
plt.figure(figsize=(8, 6))
plt.imshow(ff_history[-1], aspect='auto', cmap='gray_r',
           extent=[0, network.num_neurons, 0, network.num_inputs])
plt.colorbar(label="Feedforward Weight Strength")
plt.xlabel("Network Neuron")
plt.ylabel("Input Neuron")
plt.title("Final Feedforward Synaptic Weight Distribution")
plt.show()

# Plot recurrent weights
plt.figure(figsize=(8, 6))
plt.imshow(recur_history[-1], aspect='auto', cmap='gray',
           extent=[0, network.num_neurons, 0, network.num_neurons])
plt.colorbar(label="Recurrent Weight Strength")
plt.xlabel("Network Neuron (Postsynaptic)")
plt.ylabel("Network Neuron (Presynaptic)")
plt.title("Final Recurrent Synaptic Weight Distribution")
plt.show()