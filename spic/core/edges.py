import numpy as np


class Edge:
    def __init__(self, pre, post, n1, n2):
        self.pre_id = pre
        self.post_id = post
        self.weights = np.random.random_sample((n2, n1)) * np.sqrt(2/n1)
        self.weights = 0.122 - np.arctan(self.weights * 10) / 12

    def tick(self, input_):
        curr_out = self.weights @ input_
        # current normalization?  no for now
        return curr_out
