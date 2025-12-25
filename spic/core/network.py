from layers import DynamicBiasLayer, OutputLayer, InputLayer
from edges import Edge
from rules import SPiCRule
from collections import defaultdict
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.input_layers = []
        self.output_layers = []
        self.edges = []
        self.adj = defaultdict(list)

    def reset_state(self):
        for layer in self.layers:
            if isinstance(layer, DynamicBiasLayer):
                layer.bias = np.zeros_like(layer.bias)

    def connect(self, pre_layer, post_layer):
        if pre_layer not in self.layers:
            self.layers.append(pre_layer)
            if isinstance(pre_layer, InputLayer):
                self.input_layers.append(pre_layer)
        if post_layer not in self.layers:
            self.layers.append(post_layer)
            if isinstance(post_layer, OutputLayer):
                self.output_layers.append(post_layer)

        self.edges.append(Edge(pre_layer.id, post_layer.id,
                          pre_layer.n, post_layer.n))

        self.adj[pre_layer.id].append(post_layer.id)

    # should implement with binary search in future?

    def get_input_layer(self, id):
        for layer in self.input_layers:
            if layer.id == id:
                return layer

    def get_output_layer(self, id):
        for layer in self.output_layers:
            if layer.id == id:
                return layer

    def get_layer(self, id):
        for layer in self.layers:
            if layer.id == id:
                return layer

    def tick(self, in_, targets=None):
        # update inputs for all InputLayer objects
        for id, x in in_.items():
            layer = self.get_layer(id)
            layer.in_ = np.append(layer.in_, np.expand_dims(x, axis=0), axis=0)

        # update targets for all OutputLayer objects
        if targets is not None:
            for id, y in targets.items():
                self.get_output_layer(id).target = y
        else:
            for layer in self.output_layers:
                layer.target = np.zeros_like(layer.target)

        # one tick layer-level
        for layer in self.layers:
            layer.tick()

            # update per layer
            if targets is not None:
                SPiCRule.update_layer_level(layer)

        # one tick edge-level
        for id, edge in enumerate(self.edges):
            pre = edge.pre_id
            post = edge.post_id
            pre_layer = self.get_layer(pre)
            post_layer = self.get_layer(post)

            out = pre_layer.spikes
            curr = edge.tick(out)
            post_layer.in_ = np.append(self.get_layer(post).in_,
                                       np.expand_dims(curr, axis=0),
                                       axis=0)

            # update per edge
            if targets is not None:
                SPiCRule.update_edge_level(pre_layer, post_layer, edge,
                                           len(self.adj[pre]))

        out = {}
        for i in self.output_layers:
            out[i.id] = np.stack(i.curr_hist, axis=0).mean(axis=0)

        # analysis on out?
        return out
