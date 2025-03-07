import numpy as np
import matplotlib.pyplot as plt

class STDP_Network:
    def __init__(self, num_neurons=250, num_poisson=250, num_inputs=1000, dt=0.1,
                 tau_pre=20.0, tau_post=20.0, tau_m=20.0, V_rest=-74.0, V_reset=-60.0,
                 V_thresh=-54.0, C_m=0.9, g_leak=0.2, tau_s=5.0,
                 A_plus_ff=0.005, A_minus_ff=0.005, A_plus_recur=0.001, A_minus_recur=0.001,
                 B_ff=1.06, B_recur=1.04, g_max=0.02, poisson_rate=10, stimulus_width=50,
                 mean_stim_time=100.0, R0=10.0, R1=80.0, sigma=None):
        """
        Initialize the network with LIF neuron dynamics and STDP learning.

        Parameters:
        - num_neurons: Number of postsynaptic (network) neurons.
        - num_inputs: Number of presynaptic (input) neurons.
        - dt: Time step (ms).
        - tau_pre, tau_post: Time constants for the STDP traces (ms).
        - tau_m: Membrane time constant (ms).
        - V_rest: Resting membrane potential (mV).
        - V_reset: Reset potential after a spike (mV).
        - V_thresh: Threshold potential for spiking (mV).
        - C_m: Membrane capacitance.
        - g_leak: Leak conductance.
        - tau_s: Synaptic time constant (ms) for decay of conductance.
        - A_plus_ff, A_minus_ff: STDP learning rates for feedforward potentiation/depression.
        - A_plus_recur, A_minus_recur: STDP learning rates for recurrent potentiation/depression.
        - B_ff, B_recur: Scaling factors for feedforward and recurrent updates.
        - g_max: Maximum synaptic weight.
        - poisson_rate: Firing rate for dedicated Poisson neurons.
        - stimulus_width: Width parameter for the initial weight distribution.
        - mean_stim_time: Mean duration of a stimulus interval (ms) for input spike generation.
        - R0: Baseline firing rate (Hz) for input neurons.
        - R1: Additional firing rate amplitude (Hz) for input neurons.
        - sigma: Standard deviation for the Gaussian profile of firing rates.
                 If None, stimulus_width is used.
        """
        self.num_neurons = num_neurons    # Number of network neurons
        self.num_inputs = num_inputs      # Number of input neurons
        self.dt = dt
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # LIF neuron parameters
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_thresh = V_thresh
        self.C_m = C_m
        self.g_leak = g_leak
        self.tau_s = tau_s

        # Feedforward STDP parameters
        self.A_plus_ff = A_plus_ff
        self.A_minus_ff = A_minus_ff
        self.B_ff = B_ff

        # Recurrent STDP parameters
        self.A_plus_recur = A_plus_recur
        self.A_minus_recur = A_minus_recur
        self.B_recur = B_recur

        self.g_max = g_max

        # Parameters for input spike generation
        self.mean_stim_time = mean_stim_time  # Mean stimulus duration (ms)
        self.R0 = R0  # Baseline firing rate (Hz)
        self.R1 = R1  # Additional rate amplitude (Hz)
        self.sigma = sigma if sigma is not None else stimulus_width  # Standard deviation

        # Initialize feedforward weights with a Gaussian distribution
        # Define a center for the Gaussian peak (here using index 700)
        center_neuron = 350
        weight_distribution = np.exp(-((np.arange(self.num_inputs) - center_neuron)**2) / (2 * stimulus_width**2))
        weight_distribution = weight_distribution / np.max(weight_distribution)  # Normalize the distribution

        # Initialize feedforward weights with random values scaled by the Gaussian profile
        self.ff_weights = np.random.uniform(0.6, 1.0, (self.num_inputs, self.num_neurons)) * weight_distribution[:, None]

        # Create a sparsity mask (20% connectivity) and apply it to the feedforward weights
        self.mask = (np.random.rand(self.num_inputs, self.num_neurons) < 0.2).astype(int)
        self.ff_weights *= self.mask

        # Initialize recurrent weights (network-to-network) to zeros
        self.recur_weights = np.zeros((num_neurons, num_neurons))

        # Dedicated Poisson neurons for background input to each network neuron
        self.poisson_rate = poisson_rate
        self.g_ex = np.zeros(num_neurons)  # Excitatory conductance for network neurons

        # Initialize membrane potentials at the resting potential
        self.V_mem = np.full(num_neurons, V_rest)
        self.spike_train = np.zeros(num_neurons)  # To record spikes for the current time step

        # (Not used in the new input generation) Preferred stimulus positions for input neurons
        self.preferred_stimulus = np.linspace(0, 1000, num_inputs)
        self.current_stimulus = None

        # Initialize STDP traces for feedforward connections (one trace per connection)
        # Each feedforward connection goes from an input neuron (i) to a network neuron (j)
        self.pre_trace_feed = np.zeros((self.num_inputs, self.num_neurons))
        self.post_trace_feed = np.zeros((self.num_inputs, self.num_neurons))

        # Initialize STDP traces for recurrent connections (one trace per connection)
        self.pre_trace_recur = np.zeros((self.num_neurons, self.num_neurons))
        self.post_trace_recur = np.zeros((self.num_neurons, self.num_neurons))

    def generate_input_spike_array(self, sim_time):
        """
        Generate a binary spike array for input neurons with time-varying firing rates.

        Parameters:
        - sim_time (float): Total simulation time in milliseconds.

        Returns:
        - spike_array (numpy.ndarray): A binary matrix of shape (num_steps, num_inputs) where
          each element indicates whether an input neuron fired (1) or not (0) at that timestep.
        """
        # Determine the number of simulation steps based on dt
        num_steps = int(sim_time / self.dt)
        spike_array = np.zeros((num_steps, self.num_inputs), dtype=int)
        current_step = 0

        # Loop over the simulation time and generate spikes in intervals
        while current_step < num_steps:
            # Sample an interval duration from an exponential distribution (ms)
            interval_duration = np.random.exponential(scale=self.mean_stim_time)
            interval_steps = max(1, int(interval_duration / self.dt))
            end_step = min(num_steps, current_step + interval_steps)

            # Randomly choose a stimulus position in the range [1, 1000]
            s = np.random.uniform(1, 1000)
            a_vals = np.arange(self.num_inputs)

            # Compute firing rates using a Gaussian profile with wrap-around components
            rates = self.R0 + self.R1 * (
                np.exp(-((s - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                np.exp(-((s + 1000 - a_vals) ** 2) / (2 * self.sigma ** 2)) +
                np.exp(-((s - 1000 - a_vals) ** 2) / (2 * self.sigma ** 2))
            )
            # Convert firing rates (in Hz) to spike probability per timestep (dt in ms)
            p = rates * self.dt / 1000.0

            # For each time step in the current interval, generate spikes using a Poisson process
            for t in range(current_step, end_step):
                spike_array[t, :] = (np.random.rand(self.num_inputs) < p).astype(int)

            current_step = end_step

        return spike_array

    def decay_traces(self):
        """
        Decay all STDP traces for both feedforward and recurrent connections.
        """
        decay_pre = np.exp(-self.dt / self.tau_pre)
        decay_post = np.exp(-self.dt / self.tau_post)
        self.pre_trace_feed *= decay_pre
        self.post_trace_feed *= decay_post
        self.pre_trace_recur *= decay_pre
        self.post_trace_recur *= decay_post

    def update_poisson_input(self):
        """
        Update the excitatory conductance from dedicated Poisson neurons.
        Each network neuron receives a background excitatory input.
        """
        poisson_spikes = np.random.rand(self.num_neurons) < (self.poisson_rate * self.dt / 1000.0)
        self.g_ex = 0.5 * poisson_spikes.astype(float)  # Scaling factor adjustable as needed

    def update_membrane_potential(self, pre_spikes_feed, pre_spikes_recur):
        """
        Update the membrane potentials of network neurons using a revised LIF dynamics update.

        This method includes:
        - Exponential decay of excitatory conductance (g_ex).
        - Increases in g_ex due to feedforward and recurrent spikes.
        - A standard integration step for the membrane potential.

        Parameters:
        - pre_spikes_feed: Boolean array (length=num_inputs) for feedforward input spikes.
        - pre_spikes_recur: Boolean array (length=num_neurons) for recurrent spikes.
        """
        # Decay the excitatory conductance using an exponential decay factor
        self.g_ex *= np.exp(-self.dt / self.tau_s)

        # Update g_ex based on feedforward input spikes
        # The dot product sums the contributions from each input neuron to each network neuron.
        self.g_ex += np.dot(pre_spikes_feed, self.ff_weights) * 3  # Multiplication factor to adjust scaling

        # Update g_ex based on recurrent spikes from the network
        self.g_ex += np.dot(pre_spikes_recur, self.recur_weights) * 3

        # Calculate the synaptic current: difference between excitatory drive and leak current
        synaptic_current = self.g_ex - self.g_leak * (self.V_mem - self.V_rest)

        # Improved LIF update:
        # Using: dV = dt / C_m * ( -g_leak*(V - V_rest) + g_ex )
        dV = (self.dt / self.C_m) * ( -self.g_leak * (self.V_mem - self.V_rest) + self.g_ex )
        self.V_mem += dV

        # Detect neurons that have reached or exceeded the threshold
        spike = self.V_mem >= self.V_thresh
        # Reset membrane potentials of spiking neurons and record the spike
        self.V_mem[spike] = self.V_reset
        self.spike_train[spike] = 1

    def update_pre_feedforward(self, pre_spikes):
        """
        Update feedforward weights based on presynaptic spikes.
        Applies depression: if an input neuron fires, its outgoing synapses are weakened
        in proportion to the corresponding post-synaptic trace.

        Parameters:
        - pre_spikes: Boolean array of length num_inputs indicating input neurons that fired.
        """
        # Update the pre-synaptic trace for all outgoing connections where the input neuron spiked
        self.pre_trace_feed[pre_spikes, :] += 1

        # For each input neuron that spiked, decrease the weight by a factor proportional to the post trace
        for i in np.where(pre_spikes)[0]:
            self.ff_weights[i, :] -= self.B_ff * self.A_minus_ff * self.post_trace_feed[i, :] * self.mask[i, :]

        # Enforce the connectivity mask and clip weights to the allowed range
        self.ff_weights *= self.mask
        self.ff_weights = np.clip(self.ff_weights, 0, self.g_max)

    def update_post_feedforward(self, post_spikes):
        """
        Update feedforward weights based on postsynaptic spikes.
        Applies potentiation: if a network neuron fires, all incoming synapses are strengthened
        according to the pre-synaptic trace.

        Parameters:
        - post_spikes: Boolean array of length num_neurons indicating which network neurons fired.
        """
        # Update the post-synaptic trace for all connections targeting spiking network neurons
        self.post_trace_feed[:, post_spikes] += 1

        # For each network neuron that spiked, increase the weight by a factor proportional to the pre trace
        for j in np.where(post_spikes)[0]:
            self.ff_weights[:, j] += self.B_ff * self.A_plus_ff * self.pre_trace_feed[:, j] * self.mask[:, j]

        # Enforce the connectivity mask and clip weights to the valid range
        self.ff_weights *= self.mask
        self.ff_weights = np.clip(self.ff_weights, 0, self.g_max)

    def update_recurrent_weights(self, pre_spikes, post_spikes):
        """
        Update recurrent weights based on STDP rules for network neurons.
        Depression occurs when a presynaptic neuron fires, and potentiation occurs when a postsynaptic neuron fires.

        Parameters:
        - pre_spikes: Boolean array (length=num_neurons) for presynaptic network spikes.
        - post_spikes: Boolean array (length=num_neurons) for postsynaptic network spikes.
        """
        # For recurrent connections, update the pre-synaptic trace for spiking neurons
        self.pre_trace_recur[pre_spikes, :] += 1
        # For each spiking presynaptic neuron, decrease its outgoing weights based on the post trace
        for i in np.where(pre_spikes)[0]:
            self.recur_weights[i, :] -= self.B_recur * self.A_minus_recur * self.post_trace_recur[i, :]

        # Update the post-synaptic trace for connections targeting spiking neurons
        self.post_trace_recur[:, post_spikes] += 1
        # For each spiking postsynaptic neuron, increase its incoming weights based on the pre trace
        for j in np.where(post_spikes)[0]:
            self.recur_weights[:, j] += self.B_recur * self.A_plus_recur * self.pre_trace_recur[:, j]

        # Clip recurrent weights so that they stay within [0, g_max]
        self.recur_weights = np.clip(self.recur_weights, 0, self.g_max)

    def simulate(self, T=1000, feed_pre_times=None, feed_post_times=None, recur_pre_times=None, recur_post_times=None):
        """
        Run the simulation for T time steps.

        Parameters:
        - T: Total number of simulation time steps.
        - feed_pre_times: List of timesteps at which to update feedforward pre-synaptic weights.
        - feed_post_times: List of timesteps at which to update feedforward post-synaptic weights.
        - recur_pre_times: (Optional) Timesteps for recurrent pre-synaptic updates.
        - recur_post_times: (Optional) Timesteps for recurrent post-synaptic updates.

        Returns:
        - weights_history_ff: History of feedforward weights over time.
        - weights_history_recur: History of recurrent weights over time.
        """
        # Set default update times if not provided
        if feed_pre_times is None:
            feed_pre_times = [10, 50, 120]
        if feed_post_times is None:
            feed_post_times = [15, 55, 130]
        if recur_pre_times is None:
            recur_pre_times = []  # Not explicitly used in this implementation
        if recur_post_times is None:
            recur_post_times = []  # Not explicitly used in this implementation

        # Generate the input spike array for the entire simulation duration
        sim_time_ms = T * self.dt
        input_spike_array = self.generate_input_spike_array(sim_time_ms)

        # Lists to store weight histories for analysis
        weights_history_ff = []
        weights_history_recur = []

        # Record spike times for event plotting (raster plot)
        spike_times = []  # List of tuples (timestep, neuron index)

        # Run the simulation for T time steps
        for t in range(T):
            # Decay the STDP traces at each timestep
            self.decay_traces()
            # Update background excitation from Poisson neurons
            self.update_poisson_input()

            # Get input spikes for the current timestep from the pre-generated array
            input_spikes = input_spike_array[t, :]

            # Reset the network spike train for the current timestep
            self.spike_train.fill(0)

            # Convert input spikes to a boolean array for STDP updates
            pre_spikes_feed = input_spikes.astype(bool)
            # For recurrent connections, use the current network spike train (from the previous update)
            pre_spikes_recur = self.spike_train.astype(bool)

            # Update membrane potentials using the current feedforward and recurrent spikes
            self.update_membrane_potential(pre_spikes_feed, pre_spikes_recur)

            # Determine which network neurons spiked in this timestep
            post_spikes = self.spike_train.astype(bool)

            # Record the spike times (for each neuron that spiked)
            for neuron_index in np.where(post_spikes)[0]:
                spike_times.append((t, neuron_index))

            # Update feedforward weights at specified timesteps
            if t in feed_pre_times:
                self.update_pre_feedforward(pre_spikes_feed)
            if t in feed_post_times:
                self.update_post_feedforward(post_spikes)

            # Update recurrent weights based on the network spikes
            self.update_recurrent_weights(pre_spikes_recur, post_spikes)

            # Save the current weight matrices for later analysis
            weights_history_ff.append(self.ff_weights.copy())
            weights_history_recur.append(self.recur_weights.copy())

        # Convert histories to numpy arrays
        weights_history_ff = np.array(weights_history_ff)
        weights_history_recur = np.array(weights_history_recur)

        # Plot a raster plot of network spikes for visualization
        self.plot_spike_events(spike_times, T)

        return weights_history_ff, weights_history_recur

    def plot_spike_events(self, spike_times, T):
        """
        Create an event plot (raster plot) of the network spike times.

        Parameters:
        - spike_times: List of tuples (timestep, neuron index) for each spike.
        - T: Total number of simulation timesteps.
        """
        # Organize spike times per neuron
        spikes_per_neuron = {i: [] for i in range(self.num_neurons)}
        for t, neuron in spike_times:
            spikes_per_neuron[neuron].append(t)

        # Prepare data for the event plot (list of lists, each for one neuron)
        event_data = [spikes_per_neuron[i] for i in range(self.num_neurons)]

        plt.figure(figsize=(10, 6))
        plt.eventplot(event_data, colors='black')
        plt.xlabel("Time (ms)")
        plt.ylabel("Network Neuron")
        plt.title("Raster Plot of Network Spikes")
        plt.xlim(0, T)
        plt.show()

# ----------------- Run the Simulation ----------------- #

# Create an instance of the STDP network
network = STDP_Network()

# Run the simulation with T timesteps (e.g., T=200 for a longer simulation)
ff_history, recur_history = network.simulate(T=200)

# Plot the final feedforward weight distribution
plt.figure(figsize=(8, 6))
plt.imshow(ff_history[-1], aspect='auto', cmap='gray_r',
           extent=[0, network.num_neurons, 0, network.num_inputs])
plt.colorbar(label="g/g_max")
plt.xlabel("Network Neuron")
plt.ylabel("Input Neuron")
plt.title("Final Feedforward Synaptic Weight Distribution")
plt.show()

# Plot the final recurrent weight distribution
plt.figure(figsize=(8, 6))
plt.imshow(recur_history[-1], aspect='auto', cmap='gray',
           extent=[0, network.num_neurons, 0, network.num_neurons])
plt.colorbar(label="Recurrent Weight Strength")
plt.xlabel("Postsynaptic Neuron")
plt.ylabel("Presynaptic Neuron")
plt.title("Final Recurrent Synaptic Weight Distribution")
plt.show()