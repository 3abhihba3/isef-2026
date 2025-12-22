from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# add a refractory period: done

DELTA_BIAS = 0.2
DELTA_WEIGHT = 0.008

np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.9g" % x))

rng = np.random.default_rng()


class Edge:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        self.w = np.exp(np.random.rand(
            self.post.n, self.pre.n) * 4) / np.e**4

    def forward(self, x):
        out = self.w @ x  # (post.n,)
        self.post.forward(out)


class Node:
    def __init__(self, n, t=20, decay=0.7, threshold=0.4, reset=-0.5, refrac_steps=4, inhib=0.1):
        self.n = n
        self.t = t
        self.inhib = inhib
        self.spike_amplitudes = np.ones(self.n, dtype=np.float32)
        self.spike_amplitudes[np.random.rand(self.n) < self.inhib] = -1
        self.edges = []  # outgoing edges
        self.target = None
        self.local_error = np.zeros(n)

        self.spikes_out = deque([np.zeros(self.n, dtype=bool)], maxlen=t)
        self.curr_in = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)
        self.curr_tot = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)
        self.potential = deque([np.zeros(self.n, dtype=np.float32)], maxlen=t)
        self.refrac_hist = deque([np.ones(self.n, dtype=bool)], maxlen=t)

        self.decay = decay
        self.threshold = threshold
        self.reset = reset
        self.bias = np.zeros(self.n, dtype=np.float32)

        self.refrac_steps = refrac_steps
        self.refrac_count = np.zeros(self.n, dtype=np.int32)

    def connect(self, next_nodes):
        for nd in next_nodes:
            self.edges.append(Edge(self, nd))

    def forward(self, syn_input):
        # forward computation
        syn_input = np.asarray(syn_input, dtype=np.float32)

        if len(self.curr_in) > 1:
            self.bias -= self.curr_in[-1] - self.curr_in[-2]
        self.update_curr(syn_input)

        old_u = self.potential[-1]
        new_u = self.decay * old_u + self.curr_tot[-1]

        can_spike = self.refrac_count == 0
        self.refrac_hist.append(can_spike)

        spikes = can_spike & (new_u > self.threshold)

        self._add_pot(new_u.copy())

        new_u = new_u.copy()
        new_u[spikes] = self.reset

        self._add_out(spikes.astype(bool))

        self.refrac_count = np.maximum(self.refrac_count - 1, 0)
        self.refrac_count[spikes] = self.refrac_steps

        spikes_float = spikes.astype(np.float32)
        spikes_float *= self.spike_amplitudes
        for e in self.edges:
            e.forward(spikes_float)

        if not self.n > 500:
            self.update()

    def _h_prime(self, u):
        u = np.asarray(u, dtype=np.float32)
        width = 9.0
        x = (u - self.threshold) / width
        grad = np.maximum(0.0, 1.0 - np.abs(x)) / width
        return grad.astype(np.float32)

    def update(self):
        if self.target is not None:
            self.local_error = np.mean(self.curr_in, axis=0) - self.target
            # print(self.local_error)
        else:
            self.local_error = -self.bias.copy()

        if not self.edges:
            return

        update_bias_list = np.zeros(
            (len(self.edges), self.n), dtype=np.float32
        )

        for i, e in enumerate(self.edges):
            post = e.post
            update_bias_list[i] = self._get_update_bias(e, post)
            e.w += self.get_update_w(e, post)
            np.clip(e.w, 0, 2)

        k = np.mean(update_bias_list, axis=0)
        self.bias += k
        np.clip(self.bias, -1, 1)

    def get_update_w(self, edge, post):
        post_in_hist = np.stack(post.curr_in, axis=0).astype(
            np.float32)    # (Tpost, post.n)
        post_tot_hist = np.stack(post.curr_tot, axis=0).astype(
            np.float32)

        Tpost = min(len(post_in_hist), len(post_tot_hist))
        post_in_hist = post_in_hist[-Tpost:]
        post_tot_hist = post_tot_hist[-Tpost:]

        pre_spk_hist = np.stack(self.spikes_out, axis=0).astype(
            np.float32)

        T = min(len(pre_spk_hist), Tpost)
        pre_spk_hist = pre_spk_hist[-T:]
        post_in_hist = post_in_hist[-T:]
        post_tot_hist = post_tot_hist[-T:]

        avg_in = post_in_hist.mean(axis=0)
        y = post_tot_hist.mean(axis=0) if (
            post.target is None) else post.target
        dE_davg_in = post.local_error.copy()  # (post.n, )

        dE_di = (dE_davg_in / float(T))  # (post.n, )

        dE_dW = np.mean(dE_di[:, None] * pre_spk_hist[:, None, :],
                        axis=0).astype(np.float32)  # (post.n, pre.n)
        return (-DELTA_WEIGHT * dE_dW).astype(np.float32)

    def _get_update_bias(self, edge, post):                               # BIAS UPDATE
        post_in_hist = np.stack(post.curr_in, axis=0).astype(
            np.float32)    # (t, post.n)
        post_tot_hist = np.stack(post.curr_tot, axis=0).astype(
            np.float32)  # (t, post.n)

        Tpost = min(len(post_in_hist), len(post_tot_hist))
        post_in_hist = post_in_hist[-Tpost:]
        post_tot_hist = post_tot_hist[-Tpost:]

        avg_in = post_in_hist.mean(axis=0)  # (post.n,)
        y = post_tot_hist.mean(axis=0) if (
            post.target is None) else post.target
        # (post.n,)
        dE_davg_in = post.local_error.copy()  # (post.n, )
        dE_di_t = (dE_davg_in / float(Tpost))  # (post.n, )

        # (pre.n, post.n) @ (post.n, ) = (pre.n, )
        g_spike_pre = (edge.w.T @ dE_di_t)

        pot_hist = np.stack(self.potential, axis=0)  # (t, pre.n)
        Tpre = pot_hist.shape[0]

        dp_db = np.zeros_like(pot_hist, dtype=np.float32)  # du_t/db
        ds_db = np.zeros_like(pot_hist, dtype=np.float32)  # ds_t/db

        enable_spike = np.stack(self.refrac_hist, axis=0)  # (t, pre.n)

        for t in range(Tpre):
            dp_prev = np.zeros(self.n, dtype=np.float32) if (
                t == 0) else dp_db[t - 1]
            dp_t = self.decay * dp_prev + 1.0
            dp_db[t] = dp_t
            ds_db[t] = self.spike_amplitudes * enable_spike[t] * \
                (self._h_prime(pot_hist[t]) * dp_t)

        # dE/db(t) = (dE/ds_pre) * (ds/db)(t)
        dE_db_t = ds_db * g_spike_pre  # (t, pre.n)

        delta = dE_db_t.mean(axis=0)
        return (-DELTA_BIAS * delta)

    def _add_in(self, x):
        self.curr_in.append(np.asarray(x, dtype=np.float32))

    def _add_tot(self, x):
        self.curr_tot.append(np.asarray(x, dtype=np.float32))

    def _add_out(self, x):
        self.spikes_out.append(np.asarray(x, dtype=bool))

    def _add_pot(self, x):
        self.potential.append(np.asarray(x, dtype=np.float32))

    def update_curr(self, x):
        self._add_in(x)
        self._add_tot(np.asarray(x, dtype=np.float32) + self.bias)


class Graph:
    def __init__(self, nodes=None):
        self.nodes = nodes or []

    def add_node(self, node):
        self.nodes.append(node)

    def connect_all(self):
        for i in range(len(self.nodes) - 1):
            self.nodes[i].connect([self.nodes[i + 1]])

    def step(self, x, target):
        self.nodes[-1].target = target
        self.nodes[0].forward(x)

    def predict(self, x):
        for i in x:
            self.nodes[0].forward(i)

        return np.mean(self.nodes[-1].curr_in, axis=0)

    def train(self, x, target):
        output = np.empty_like(target.shape, dtype=np.float32)

        for i, s in enumerate(x):
            self.step(s, target)
            np.append(output, self.nodes[-1].curr_in)

        return output


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


if __name__ == "__main__":

    x = Node(784, refrac_steps=0, inhib=0)
    y = Node(32, inhib=0.3)
    z = Node(32, inhib=0.2)
    w = Node(10, inhib=0.3)
    i = Node(10)

    graph = Graph([x, y, z, w, i])
    graph.connect_all()

    target = np.zeros(10, dtype=np.float32) - 1/9
    # target[0] = 200
    target[9] = 1.0

    tracked_layers = list(range(len(graph.nodes) - 1))  # 0..(L-2)
    layer_names = [f"Layer {k} (n={graph.nodes[k].n})" for k in tracked_layers]

    prev_bias = {k: graph.nodes[k].bias.copy() for k in tracked_layers}
    prev_in = {k: graph.nodes[k].curr_in[-1].copy() for k in tracked_layers}

    db_norm = {k: [] for k in tracked_layers}
    di_norm = {k: [] for k in tracked_layers}

    T = 500
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for t in range(T):
        np.random.seed()
        in_ = np.random.rand(784).astype(
            np.float32)
        in_ *= 0 / np.sum(in_)
        graph.step(in_, target)
        # print(np.linalg.norm(np.mean(graph.nodes[-3].spikes_out, axis=0)))
        bias_l2 = np.linalg.norm(graph.nodes[1].bias)
        bias_l3 = np.linalg.norm(graph.nodes[2].bias)
        bias_l4 = np.linalg.norm(graph.nodes[3].bias)
        # print(bias_l2, bias_l3, bias_l4)
        #
        print(
            np.stack(graph.nodes[-2].curr_in, axis=0).mean(axis=0), graph.nodes[-2].bias)
        a1.append(bias_l2)
        a2.append(bias_l3)
        a3.append(bias_l4)
        a4.append(
            ((np.mean(graph.nodes[-1].curr_in, axis=0) - target) ** 2).mean())
    plt.figure()
    plt.plot(a1)
    plt.plot(a2)
    plt.plot(a3)
    plt.plot(a4)

    plt.figure()
    plt.hist(graph.nodes[-2].edges[0].w, bins=5)
    plt.show()
    #
    print("FINAL RESULTS:")

    # print(np.argmax(graph.predict(10 * np.zeros(784)[:, np.newaxis])))
    print("input", np.sum(in_))
    print(graph.nodes[-2].bias)
    # print(np.stack(graph.nodes[-2].spikes_out, axis=0).astype(np.float32))
    # print(graph.nodes[-2].edges[0].w)
    # print(np.stack(graph.nodes[-2].curr_in, axis=0))

    # print(np.stack(node.spikes_out, axis=0))
    # print(np.stack(node.bias, axis=0))
    #
    # print(graph.predict(np.zeros(784)))
    print(np.mean(graph.nodes[-1].curr_in, axis=0))
    # print(np.stack(graph.nodes[-2].spikes_out, axis=0).astype(float))
