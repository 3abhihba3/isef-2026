import numpy as np
from collections import deque
from abc import ABC, abstractmethod

np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.9g" % x))


class LearningRule(ABC):
    @abstractmethod
    def update_edge_level(pre_layer, post_layer, edge, error, n):
        pass

    @abstractmethod
    def update_layer_level(layer):
        pass


class SPiCRule(LearningRule):
    min_bias = -1
    max_bias = 1

    min_curr = -0.1
    max_curr = 2

    min_weight = 0
    max_weight = 2.5

    def update_edge_level(pre_layer, post_layer, edge, n):
        if not n:
            return

        error = None
        if isinstance(post_layer, OutputLayer):
            error = np.mean(post_layer.curr_hist, axis=0) - post_layer.target
        else:
            error = -post_layer.bias

        # Bias update
        gradient = edge.weights.T @ (error /
                                     len(pre_layer.spikes_hist))

        pot_hist = np.stack(pre_layer.pot_hist, axis=0)

        potential_gradient = np.zeros_like(pot_hist)
        bias_gradient = np.zeros_like(pot_hist)

        for t in range(len(pot_hist)):
            prev_pot_grad = np.zeros(
                pre_layer.n, dtype=np.float16) if t == 0 else \
                potential_gradient[t - 1].copy()

            prev_pot_grad = (1 - DynamicBiasLayer.decay) * prev_pot_grad + 1
            potential_gradient[t] = prev_pot_grad
            bias_gradient = prev_pot_grad * pre_layer.spike_mag * \
                pre_layer.h_prime(pot_hist[t])

        pre_layer.bias += -gradient * np.mean(bias_gradient, axis=0) * 0.2 / n
        pre_layer.bias = np.clip(
            pre_layer.bias, SPiCRule.min_bias, SPiCRule.max_bias)

        # Weight update
        pass
        num = len(pre_layer.spikes_hist)
        spikes_hist = np.stack(pre_layer.spikes_hist, axis=0) / num
        gradient = np.zeros_like(edge.weights, dtype=np.float16)
        for i in spikes_hist:
            gradient += np.tile(i, (post_layer.n, 1))

        edge.weights += -0.001 * error.reshape(-1, 1) * gradient
        edge.weights = np.clip(
            edge.weights, SPiCRule.min_weight, SPiCRule.max_weight)

    def update_layer_level(layer):
        if isinstance(layer, OutputLayer):
            return
        # Bias stabilization
        # Make sure total input current never goes beyond some threshold
        curr_prev = layer.curr_hist[-1] + layer.bias
        delta_left = SPiCRule.min_curr - curr_prev
        delta_right = curr_prev - SPiCRule.max_curr
        cond = np.logical_xor(delta_left > 0, delta_right > 0)
        delta = np.maximum(delta_left, delta_right)[cond]
        layer.bias[cond] += delta * 0.01

        layer.bias = np.clip(layer.bias, SPiCRule.min_bias, SPiCRule.max_bias)

        # Should be done once per layer

# Base class for spiking network layer


class SNNLayer:
    decay = 0.2
    threshold = 0.3
    reset = 0

    def __init__(self, n):
        self.n = n
        self.pot = np.zeros(n, dtype=np.float16)

        self.in_ = np.zeros((1, n), dtype=np.float16)
        self.spikes = np.zeros(n, dtype=np.float16)

    def _reset_in_(self):
        self.in_ = np.zeros((1, self.n), dtype=np.float16)

    def tick(self):
        total_curr = np.zum(self.in_, axis=0)

        self.pot = self.pot * (1 - SNNLayer.decay) + total_curr
        spikes_bool = self.pot > SNNLayer.threshold
        self.pot[spikes_bool] = SNNLayer.reset
        self.spikes = spikes_bool.astype(np.float16)


class DynamicBiasLayer(SNNLayer):
    t = 10
    pot_max = 20.0

    def __init__(self, n):
        super().__init__(n)
        self.bias = np.zeros(n, dtype=np.float16)
        self.spike_mag = np.ones(n)

        self.pot_hist = deque(maxlen=self.t)
        self.curr_hist = deque(maxlen=self.t)
        self.spikes_hist = deque(maxlen=self.t)

    def tick(self):
        # Update current
        total_curr = np.sum(self.in_, axis=0) + self.bias
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

    def h_prime(self, x):
        width = 1.3  # testing
        u = (x - self.threshold) / width
        grad = np.maximum(0.0, 1.0 - np.abs(u))
        return grad.astype(np.float16)


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


class Edge:
    def __init__(self, pre, post, n1, n2):
        self.pre_id = pre
        self.post_id = post
        self.weights = np.exp(np.random.rand(n2, n1) * 5) / 150

    def tick(self, input_):
        curr_out = self.weights @ input_
        # current normalization?
        return curr_out


class Network:
    def __init__(self):
        # for MNIST solver
        self.layers = [
            DynamicBiasLayer(784),
            DynamicBiasLayer(32),
            DynamicBiasLayer(32),
            DynamicBiasLayer(10),
            OutputLayer(10),
        ]

        self.edges = [
            Edge(0, 1, 784, 32),
            Edge(1, 2, 32, 32),
            Edge(2, 3, 32, 10),
            Edge(3, -1, 10, 10)
        ]
        self.adj = [[1], [2], [3], [-1], []]

        self.n_out = 1

    def create_network(self):
        pass

    def tick(self, in_, targets=None):
        self.layers[0].in_ = np.append(self.layers[0].in_, in_, axis=0)
        if targets is not None:
            for id in range(self.n_out):
                target = targets[id]
                self.layers[-(id + 1)].target = target

        for layer in self.layers:
            layer.tick()

            # update per layer
            SPiCRule.update_layer_level(layer)

        for id, edge in enumerate(self.edges):
            pre = edge.pre_id
            post = edge.post_id
            pre_layer = self.layers[pre]
            post_layer = self.layers[post]

            out = pre_layer.spikes
            curr = edge.tick(out)
            post_layer.in_ = np.append(self.layers[post].in_,
                                       np.expand_dims(curr, axis=0),
                                       axis=0)

            # update per edge
            SPiCRule.update_edge_level(pre_layer, post_layer, edge,
                                       len(self.adj[pre]))

        out = np.array([np.stack(self.layers[-(id + 1)].curr_hist, axis=0).mean(axis=0)
                        for id in range(self.n_out)])

        # analysis on out?
        #
        return out
