from layers import OutputLayer, InputLayer, DynamicBiasLayer
from abc import ABC, abstractmethod
import numpy as np


# An informal interface for learning rules
class LearningRule(ABC):
    @abstractmethod
    def update_edge_level(pre_layer, post_layer, edge, error, n):
        pass

    @abstractmethod
    def update_layer_level(layer):
        pass


class SPiCRule(LearningRule):
    min_bias = -0.3
    max_bias = 0.8

    min_curr = -1
    max_curr = 1

    min_weight = 0.0
    max_weight = 2.5

    bias_lr = 0.2
    weights_lr = 0.01

    bias_decay = 0.042

    # 0 for exact same value as bias, 0.99999... for max smoothness
    bias_trace_smoothness = 0.7

    def _weights_calc(t):
        return SPiCRule.weights_lr

    def update_edge_level(pre_layer, post_layer, edge, n, t):
        if not n:
            return

        error = None
        if isinstance(post_layer, OutputLayer):
            error = np.mean(post_layer.curr_hist, axis=0) - post_layer.target
        else:
            error = -post_layer.bias_trace

        # Bias update
        if isinstance(pre_layer, DynamicBiasLayer):
            gradient = edge.weights.T @ (error /
                                         len(pre_layer.spikes_hist))

            pot_hist = np.stack(pre_layer.pot_hist, axis=0)

            potential_gradient = np.zeros_like(pot_hist)
            bias_gradient = np.zeros_like(pot_hist)

            for t in range(len(pot_hist)):
                prev_pot_grad = np.zeros(
                    pre_layer.n, dtype=np.float16) if t == 0 else \
                    potential_gradient[t - 1].copy()

                prev_pot_grad = ((1 - DynamicBiasLayer.decay)
                                 * prev_pot_grad + 1)
                potential_gradient[t] = prev_pot_grad
                bias_gradient = prev_pot_grad * pre_layer.spike_mag * \
                    DynamicBiasLayer.h_prime(pot_hist[t])

            pre_layer.bias += -gradient * \
                np.mean(bias_gradient, axis=0) * SPiCRule.bias_lr / n
            pre_layer.bias = np.clip(
                pre_layer.bias, SPiCRule.min_bias, SPiCRule.max_bias)

        # Weight update

        num = len(pre_layer.spikes_hist)
        spikes_hist = np.stack(pre_layer.spikes_hist, axis=0) / num
        gradient = np.zeros_like(edge.weights, dtype=np.float16)
        for i in spikes_hist:
            gradient += np.tile(i, (post_layer.n, 1))

        edge.weights += - \
            SPiCRule._weights_calc(t) * error.reshape(-1, 1) * gradient
        edge.weights = np.clip(
            edge.weights, SPiCRule.min_weight, SPiCRule.max_weight)

    def update_layer_level(layer):
        if isinstance(layer, OutputLayer) or isinstance(layer, InputLayer):
            return
        # Bias stabilization
        # Make sure total input current never goes beyond some threshold
        # curr_prev = layer.curr_hist[-1] + layer.bias
        # delta_left = SPiCRule.min_curr - curr_prev
        # delta_right = curr_prev - SPiCRule.max_curr
        # cond = np.logical_xor(delta_left > 0, delta_right > 0)
        # delta = np.maximum(delta_left, delta_right)[cond]
        # layer.bias[cond] += delta * 0.001
        layer.bias -= layer.bias * SPiCRule.bias_decay
        layer.bias = np.clip(layer.bias, SPiCRule.min_bias, SPiCRule.max_bias)
        layer.bias_trace *= SPiCRule.bias_trace_smoothness
        layer.bias_trace += layer.bias * (1 - SPiCRule.bias_trace_smoothness)

        # Should be done once per layer
