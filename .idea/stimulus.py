import numpy as np
import matplotlib.pyplot as plt


class STDP_Network:
    def __init__(self, num_neurons=250, num_poisson=250, num_inputs=1000, dt=1.0,
                 tau_pre=20.0, tau_post=20.0, tau_m=20.0, V_rest=-74.0, V_reset=-60.0,
                 V_thresh=-54.0, C_m=1.0, g_leak=1.0, tau_s=5.0,
                 A_plus_ff=0.005, A_minus_ff=0.005, A_plus_recur=0.001, A_minus_recur=0.001,
                 B_ff=1.06, B_recur=1.04, g_max=0.02, poisson_rate=10, stimulus_width=100):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.dt = dt
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_thresh = V_thresh
        self.C_m = C_m
        self.g_leak = g_leak
        self.tau_s = tau_s

        self.A_plus_ff = A_plus_ff
        self.A_minus_ff = A_minus_ff
        self.B_ff = B_ff

        self.A_plus_recur = A_plus_recur
        self.A_minus_recur = A_minus_recur
        self.B_recur = B_recur

        self.g_max = g_max

        self.ff_weights = np.random.rand(num_inputs, num_neurons) * g_max
        self.mask = (np.random.rand(num_inputs, num_neurons) < 0.2).astype(int)
        self.ff_weights *= self.mask

        self.recur_weights = np.zeros((num_neurons, num_neurons))
        self.poisson_rate = poisson_rate
        self.g_ex = np.zeros(num_neurons)
        self.V_mem = np.full(num_neurons, V_rest)
        self.spike_train = np.zeros(num_neurons)

        self.stimulus_width = stimulus_width
        self.preferred_stimulus = np.linspace(0, 1000, num_inputs)
        self.current_stimulus = None

        self.pre_trace_feed = np.zeros(num_inputs)
        self.post_trace_feed = np.zeros(num_neurons)

        self.pre_trace_recur = np.zeros(num_neurons)
        self.post_trace_recur = np.zeros(num_neurons)

    def decay_traces(self):
        self.pre_trace_feed *= np.exp(-self.dt / self.tau_pre)
        self.post_trace_feed *= np.exp(-self.dt / self.tau_post)
        self.pre_trace_recur *= np.exp(-self.dt / self.tau_pre)
        self.post_trace_recur *= np.exp(-self.dt / self.tau_post)

    def update_poisson_input(self):
        poisson_spikes = np.random.rand(self.num_neurons) < (self.poisson_rate * self.dt / 1000.0)
        self.g_ex = 0.096 * poisson_spikes.astype(float)

    def generate_stimulus(self, t):
        if t == 700:
            self.current_stimulus = 700
            rates = 10 + 200 * np.exp(-((self.preferred_stimulus - self.current_stimulus) ** 2) / (2 * self.stimulus_width ** 2))
        else:
            self.current_stimulus = np.random.uniform(0, 1000)
            rates = 10 + 80 * np.exp(-((self.preferred_stimulus - self.current_stimulus) ** 2) / (2 * self.stimulus_width ** 2))
        input_spikes = np.random.poisson(rates / 1000.0)
        return input_spikes

    def update_membrane_potential(self):
        synaptic_current = self.g_ex - self.g_leak * (self.V_mem - self.V_rest)
        dV = (synaptic_current / self.C_m) * (self.dt / self.tau_m)
        self.V_mem += dV

        spike = self.V_mem >= self.V_thresh
        self.V_mem[spike] = self.V_reset
        self.spike_train[spike] = 1

    def update_recurrent_weights(self, pre_spikes, post_spikes):
        self.pre_trace_recur[pre_spikes] += 1
        for i in np.where(pre_spikes)[0]:
            self.recur_weights[i, :] -= self.B_recur * self.A_minus_recur * self.post_trace_recur
        self.recur_weights = np.clip(self.recur_weights, 0, 1)

        self.post_trace_recur[post_spikes] += 1
        for j in np.where(post_spikes)[0]:
            self.recur_weights[:, j] += self.B_recur * self.A_plus_recur * self.pre_trace_recur
        self.recur_weights = np.clip(self.recur_weights, 0, 1)

    def simulate(self, T=1000):
        weights_history_ff = []
        weights_history_recur = []

        for t in range(T):
            self.decay_traces()
            self.update_poisson_input()
            _ = self.generate_stimulus(t)

            self.update_membrane_potential()
            post_spikes = self.spike_train.astype(bool)
            self.update_recurrent_weights(post_spikes, post_spikes)

            weights_history_ff.append(self.ff_weights.copy())
            weights_history_recur.append(self.recur_weights.copy())

        return np.array(weights_history_ff), np.array(weights_history_recur)


# Run the simulation
network = STDP_Network()
ff_history, recur_history = network.simulate(T=1000)

# Plot Feedforward Weight Distribution
plt.figure(figsize=(8, 6))
plt.imshow(ff_history[-1], aspect='auto', cmap='hot',
           extent=[0, network.num_neurons, 0, network.num_inputs])
plt.colorbar(label="Feedforward Weight Strength")
plt.xlabel("Network Neuron")
plt.ylabel("Input Neuron")
plt.title("Final Feedforward Synaptic Weight Distribution")
plt.show()

# Plot Recurrent Weight Distribution
plt.figure(figsize=(8, 6))
plt.imshow(recur_history[-1], aspect='auto', cmap='hot',
           extent=[0, network.num_neurons, 0, network.num_neurons])
plt.colorbar(label="Recurrent Weight Strength")
plt.xlabel("Network Neuron (Postsynaptic)")
plt.ylabel("Network Neuron (Presynaptic)")
plt.title("Final Recurrent Synaptic Weight Distribution")
plt.show()