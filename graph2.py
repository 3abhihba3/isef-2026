from collections import deque
import numpy as np

# Learning rates
DELTA_BIAS = 0.5
DELTA_WEIGHT = 0.01
np.set_printoptions(linewidth=900)


class Edge:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        # weight matrix: (post.n, pre.n) so w @ spikes_pre works
        self.w = np.random.rand(self.post.n, self.pre.n).astype(np.float32)

    def forward(self, x):
        """
        x: pre-synaptic spikes (shape (pre.n,))
        """
        out = self.w @ x  # (post.n,)
        self.post.forward(out)


class Node:
    def __init__(self, n, t=10, decay=0.67, threshold=1.0, reset=0.0):
        self.n = n
        self.t = t
        self.edges = []  # outgoing edges
        self.target = None

        # Time histories (most recent at the right). maxlen enforces window size.
        self.spikes_out = deque([np.zeros(self.n, dtype=bool)], maxlen=t)
        self.curr_in = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)
        self.curr_tot = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)
        self.potential = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)

        self.decay = decay
        self.threshold = threshold
        self.reset = reset
        self.bias = np.zeros(self.n, dtype=np.float32)

    def connect(self, next_nodes):
        """
        Connect this node to a list of post-synaptic nodes.
        """
        for nd in next_nodes:
            self.edges.append(Edge(self, nd))

    def forward(self, syn_input, update=True):
        """
        One time step of LIF dynamics.

        syn_input: synaptic input current from presynaptic spikes (shape (n,))
        """
        syn_input = np.asarray(syn_input, dtype=np.float32)

        # 1) Update currents (store input, add bias)
        self.update_curr(syn_input)

        # 2) Update membrane potential: u_t = decay * u_{t-1} + I_t
        old_u = self.potential[-1]
        new_u = self.decay * old_u + self.curr_tot[-1]

        # 3) Generate spikes
        spikes = new_u > self.threshold

        self._add_pot(new_u.copy())
        if self.n == 16:
            print(self.bias[:4])
        # 4) Reset potentials where spikes occurred
        new_u[spikes] = self.reset
        # 5) Store potential and spikes
        self._add_out(spikes.astype(bool))

        # 6) Propagate spikes forward (as float, so w @ spikes works)
        spikes_float = spikes.astype(np.float32)
        for e in self.edges:
            e.forward(spikes_float)

        self.update()

    def _f_prime(self, u):
        """
        Placeholder for derivative of any continuous nonlinearity on current.
        For now, treat it as identity => derivative 1 everywhere.
        """
        return np.ones_like(u, dtype=np.float32)

    def _h_prime(self, u):
        """
        Surrogate derivative of spike function s = H(u - threshold).

        Use a simple piecewise-linear surrogate: non-zero in a small band
        around the threshold. This is crucial for making any gradient
        signal flow through the hard threshold.
        """
        u = np.asarray(u, dtype=np.float32)
        width = 5  # width of the linear window around threshold
        x = (u - self.threshold) / width
        grad = np.maximum(0.0, 1.0 - np.abs(x)) / width
        return grad.astype(np.float32)

    def update(self):
        """
        Local update of this node's bias using all outgoing edges.

        Intuition: each outgoing synapse gives us a local “teaching signal”
        based on how its postsynaptic node is behaving. We average these
        signals to get a bias update.
        """
        if not self.edges:
            return

        update_bias_list = np.zeros(
            (len(self.edges), self.n),
            dtype=np.float32
        )
        for i, e in enumerate(self.edges):
            post = e.post
            update_bias_list[i] = self._get_update_bias(e, post)
            e.w += self.get_update_w(e, post)

        # Average contribution across outgoing edges (axis=0 is important!)
        total_update_bias = np.mean(update_bias_list, axis=0)
        self.bias += total_update_bias

    def get_update_w(self, edge, post):
        # TODO: define a local weight update rule later
        T_post = len(post.curr_in)

        curr_avg = np.stack(post.curr_in, axis=0).mean(axis=0)
        curr_y = np.stack(post.curr_tot, axis=0).mean(
            axis=0) if self.target is None else self.target

        dE_davg = curr_avg - curr_y
        dE_di_t = (dE_davg / T_post)[:, None]
        print(dE_di_t.shape)

        dE_dw = np.mean(np.asarray([dE_di_t * np.tile(t, post.n)
                        for t in self.spikes_out]), axis=0)

        print(dE_dw.shape)
        return -DELTA_WEIGHT * dE_dw

    def _get_update_bias(self, edge, post):
        """
        Compute a local gradient-based update for this node's bias using a
        single postsynaptic node `post` and the synapse `edge`.

        Structure of the logic (this is the key “intuitively correct” part):

        1. Define a local loss at the postsynaptic node:
           E_post = 0.5 * || avg(post.curr_in) - avg(post.curr_tot) ||^2

           So we want the average raw input current into the postsynaptic
           neuron to match its average total current (input + bias/other
           terms). This is a kind of local consistency objective.

        2. Compute dE/d(avg_in_post). This is straightforward.

        3. Spread that error uniformly across time at the postsynaptic node
           (a simple approximation, but keeps it local in time).

        4. Propagate that error locally through the synapse:
           i_post = W s_pre  =>  dE/ds_pre = W^T dE/di_post

        5. Convert error on our spikes into error on our bias using the
           membrane dynamics and the surrogate derivative of the spike
           function (this is where dp_t/db and ds_t/db come in).

        6. Average across time and do a small gradient-descent step.
        """
        # ---- 1. Build post-synaptic histories ----
        T_pre = len(self.potential)          # time window at pre
        T_post = len(post.curr_in)
        # T_post_tot = len(post.curr_tot)

        # (T_post_in, post.n)
        post_in_hist = np.stack(post.curr_in, axis=0)
        # (T_post_tot, post.n)
        post_tot_hist = np.stack(post.curr_tot, axis=0)

        # Use the common tail of both histories in case lengths differ
        # T_post = min(T_post_in, T_post_tot)
        # post_in_hist = post_in_hist[-T_post:]
        # post_tot_hist = post_tot_hist[-T_post:]

        # ---- 2. Local postsynaptic error ----
        curr_avg = post_in_hist.mean(axis=0)   # (post.n,)
        curr_y = post_tot_hist.mean(
            axis=0) if self.target is None else self.target

        # For E = 0.5 * ||avg_in - avg_tot||^2:
        # dE/d(avg_in) = (avg_in - avg_tot)
        dE_davg_in = (curr_avg - curr_y)    # (post.n,)

        # ---- 3. Distribute error across time at post ----
        dE_di_t = dE_davg_in / float(T_post)  # same at each t

        # ---- 4. Local backprop through synapse: i_post = W s_pre ----
        # dE/ds_pre = W^T * dE/di_post
        g_spike_pre = edge.w.T @ dE_di_t   # (pre.n,)

        # ---- 5. Compute ds_t/db at this presynaptic node ----
        pot_hist = np.stack(self.potential, axis=0)  # (T_pre, pre.n)
        dp_db = np.zeros_like(pot_hist, dtype=np.float32)
        ds_db = np.zeros_like(pot_hist, dtype=np.float32)

        for t in range(T_pre):
            if t == 0:
                dp_prev = np.zeros(self.n, dtype=np.float32)
            else:
                dp_prev = dp_db[t - 1]

            # LIF dynamics (simplified derivative):
            # u_t = decay * u_{t-1} + I_t + b
            # => d u_t / d b = decay * d u_{t-1}/d b + 1
            dp_t = self.decay * dp_prev + 1.0
            dp_db[t] = dp_t

            # Spike derivative via surrogate: s_t ~ H(u_t - theta)
            ds_db[t] = self._h_prime(pot_hist[t]) * dp_t  # elementwise

        # Error on bias at each time:
        # dE/db_j(t) = (dE/ds_pre_j) * ds_db_j(t)
        # g_spike_pre: (pre.n,)
        # ds_db      : (T_pre, pre.n)
        dE_db_t = ds_db * g_spike_pre   # broadcast -> (T_pre, pre.n)

        # ---- 6. Average over time & step in negative gradient direction ----
        delta = dE_db_t.mean(axis=0)       # (pre.n,)
        return -DELTA_BIAS * delta         # gradient descent

    def _add_in(self, x):
        self.curr_in.append(np.asarray(x, dtype=np.float32))

    def _add_tot(self, x):
        self.curr_tot.append(np.asarray(x, dtype=np.float32))

    def _add_out(self, x):
        self.spikes_out.append(np.asarray(x, dtype=bool))

    def _add_pot(self, x):
        self.potential.append(np.asarray(x, dtype=np.float32))

    def update_curr(self, x):
        """
        Store raw synaptic input and compute total current including bias.
        """
        self._add_in(x)
        total = np.asarray(x, dtype=np.float32) + self.bias
        self._add_tot(total)


class Graph:
    def __init__(self, nodes=None):
        self.nodes = nodes or []

    def add_node(self, node):
        self.nodes.append(node)

    def connect_all(self):  # temporary
        for i in range(len(self.nodes) - 1):
            self.nodes[i].connect([self.nodes[i + 1]])

    def step(self, x, target):
        self.nodes[-2].target = target
        self.nodes[0].forward(x)


x = Node(784)
y = Node(16)
z = Node(16)
w = Node(10)
i = Node(10)
graph = Graph([x, y, z, w, i])
target = np.zeros(10, dtype=np.float32)
target[[1, 3, 5]] = 1.0
graph.connect_all()
for i in range(0, 40):
    graph.step(np.random.rand(784) * 0.5, target)
