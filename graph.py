from collections import deque
import numpy as np
# TODO: fix bugs, write graph (add scheduling for [node] -> node events, global tick),
DELTA_BIAS = 0.1
DELTA_WEIGHT = 0.01


class edge:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        self.w = np.random.rand(self.post.n, self.pre.n)

    def forward(self, x):
        out = np.matmul(self.w, x)
        self.post.forward(out)


class node:
    def __init__(self, n):
        self.n = n
        self.t = 20
        self.edges = []

        self.spikes_out = deque([np.zeros(self.n, dtype=bool)])
        self.curr_in = deque([np.zeros(self.n, dtype=np.float16)])
        self.curr_tot = deque([np.zeros(self.n, dtype=np.float16)])
        self.potential = deque([np.zeros(self.n, dtype=np.float16)])

        self.decay = 0.67
        self.threshold = 1.0
        self.reset = 0.0
        self.bias = np.zeros(self.n)

    def connect(self, next):
        for node in next:
            self.edges.append(edge(self, node))

    def forward(self, x):
        old = self.potential[-1]
        new = old * self.decay
        self.update_curr(x)
        new += self.curr_tot[-1]
        spikes = new > self.threshold
        new[spikes] = self.reset
        self._add_pot(new)
        self._add_out(new)

        for e in self.edges:
            e.forward(spikes)

    def _f_prime(self, u):
        pass

    def _h_prime(self, u):
        pass

    def _update(self):
        # update self and all edges
        update_bias_list = np.zeros(
            (len(self.edges), self.n), dtype=np.float16)
        for i in range(len(update_bias_list)):
            edge = self.edges[i]
            post = edge.post
            update_bias_list[i] = self._get_update_bias(edge, post)

        total_update_bias = np.mean(update_bias_list)
        self.bias += total_update_bias

    def _update_w(self, edge, post):
        pass  # don't do this for now

    def _get_update_bias(self, edge, post):
        ds_t_wrt_b = np.zeros((len(self.potential), self.n), dtype=np.float16)
        dp_t_wrt_b = np.zeros((len(self.potential), self.n), dtype=np.float16)
        dp_t_wrt_b[1] = np.ones(self.n, dtype=np.float16)
        ds_t_wrt_b[1] = dp_t_wrt_b + self._h_prime(self.potential[1])
        for i, val in ds_t_wrt_b:
            if i <= 1:
                continue
            dp_t_wrt_b[i] = 1 + dp_t_wrt_b[i - 1] - ds_t_wrt_b[i - 1]
            ds_t_wrt_b = self._h_prime * dp_t_wrt_b

        curr_avg = np.mean(post.curr_in, axis=0)
        curr_y = np.mean(post.curr_tot, axis=0)
        at = np.zeros((len(self.potential), self.n), dtype=np.float16)
        dE_wrt_avg = 2(curr_avg - curr_y) / self.n
        for t in range(len(at)):
            dE_wrt_i = dE_wrt_avg / len(self.potential)
            dE_wrt_s_t = dE_wrt_i[:, None] * edge.w
            dE_wrt_b = np.matmul(dE_wrt_s_t, ds_t_wrt_b[t])
            at[t] = dE_wrt_b

        delta = np.mean(at, axis=0)
        return delta * DELTA_BIAS

    def _add_in(self, x):
        self.curr_in.append(x)
        if len(self.curr_in) > self.t:
            self.curr_in.popLeft()

    def _add_tot(self, x):
        self.curr_tot.append(x)
        if len(self.curr_tot) > self.t:
            self.curr_tot.popLeft()

    def _add_out(self, x):
        self.spikes_out.append(x)
        if len(self.spikes_out) > self.t:
            self.spikes_out.popLeft()

    def _add_pot(self, x):
        self.potential.append(x)
        if len(self.potential) > self.t:
            self.potential.popLeft()

    def update_curr(self, x):
        self._add_in(x)
        x += self.bias
        self.add_tot(x)


class graph:
    def __init__(self):
        pass
