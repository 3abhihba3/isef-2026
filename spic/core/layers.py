import numpy as np
from collections import deque


class SNNLayer:
    decay = 0.2
    threshold = 1
    reset = 0
    num = 0

    def __init__(self, n):
        self.n = n

        SNNLayer.num += 1
        self.id = SNNLayer.num

        self.pot = np.zeros(n, dtype=np.float16)

        self.spikes_hist = deque(maxlen=self.t)

        self.in_ = np.zeros((1, n), dtype=np.float16)
        self.spikes = np.zeros(n, dtype=np.float16)

    def _reset_in_(self):
        self.in_ = np.zeros((1, self.n), dtype=np.float16)

    def update_input(self, curr):
        self.in_ = np.append(self.in_, np.expand_dims(curr, axis=0), axis=0)

    def tick(self):
        total_curr = np.sum(self.in_, axis=0)

        self.pot = self.pot * (1 - SNNLayer.decay) + total_curr
        spikes_bool = self.pot > SNNLayer.threshold
        self.pot[spikes_bool] = SNNLayer.reset
        self.spikes = spikes_bool.astype(np.float16)
        self.spikes_hist.append(self.spikes)
        self._reset_in_()


class DynamicBiasLayer(SNNLayer):
    t = 10
    pot_max = 20.0

    def __init__(self, n):
        super().__init__(n)
        self.bias = np.zeros(n, dtype=np.float16)
        self.offset = np.zeros(n, dtype=np.float16)
        self.spike_mag = np.random.binomial(1, 0.5, n)
        # updated in SPiCRule for now
        self.bias_trace = np.zeros_like(self.bias)

        self.pot_hist = deque(maxlen=self.t)
        self.curr_hist = deque(maxlen=self.t)
        self.spikes_hist = deque(maxlen=self.t)

    def tick(self):
        # Update current
        total_curr = np.sum(self.in_, axis=0) + self.bias + 0.17
        self.curr_hist.append(total_curr)

        # Update potential
        self.pot = self.pot * (1 - self.decay) + total_curr
        self.pot = np.clip(self.pot, self.reset, self.pot_max)
        self.pot_hist.append(self.pot.copy())

        # Calculate spikes
        spike_bool = self.pot > self.threshold
        self.spikes = spike_bool.astype(np.float16) * self.spike_mag
        self.spikes_hist.append(self.spikes.copy())
        # reset potential and input current
        self.pot[spike_bool] = self.reset
        self._reset_in_()

    def h_prime(x):
        width = 1.3  # testing
        u = (x - DynamicBiasLayer.threshold) / width
        grad = np.maximum(0.0, 1.0 - np.abs(u))
        return grad.astype(np.float16)

    def h(x):
        new = x.copy()
        new[new < DynamicBiasLayer.threshold] = 0
        new[new != 0] = 1
        return new


class OutputLayer(SNNLayer):
    t = 15

    def __init__(self, n):
        super().__init__(n)
        self.target = np.zeros(n, dtype=np.float16)

        self.curr_hist = deque(maxlen=self.t)

    def tick(self):
        total_curr = np.sum(self.in_, axis=0)
        self.curr_hist.append(total_curr)
        self._reset_in_()


class InputLayer(SNNLayer):
    t = 5

    def __init__(self, n):
        super().__init__(n)
