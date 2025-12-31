import matplotlib.pyplot as plt
from network import Network
from layers import DynamicBiasLayer, InputLayer, OutputLayer
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.9g" % x))

# a super barebones test to see if the learning rule works across at least one hidden layer.
# basically just try to model the function y=x with the input x as a one-hot vector
# no batch size so one at a time


l1 = InputLayer(10)
l2 = DynamicBiasLayer(16)
l4 = OutputLayer(10)

x = Network()
x.connect(l1, l2)
x.connect(l2, l4)
# plt.hist(x.edges[1].weights)
# plt.show()
input_ = {
    l1.id: np.zeros(10)
}

target = {
    l4.id: np.zeros(10)
}

L = []


def train(n, plot=False):
    for i in range(0, n):
        a = np.zeros(10)
        n = i % 10
        a[n] = 1
        input_[l1.id] = a
        target[l4.id] = a
        print("iteration", i, "k=", n)
        x.reset_state()
        bias = []
        spikes = [[] for i in range(l2.n)]
        l = 0
        for i in range(500):
            k = x.tick(input_, target)
            bias.append(l2.bias)
            l += ((k[l4.id] - target[l4.id]) ** 2).mean(axis=0) / 500
            s = l2.spikes
            s[s > 0] = i
            for i, a in enumerate(s):
                if a != 0:
                    spikes[i].append(a)
        if plot:
            plt.eventplot(spikes)
            plt.title(f"{n}: LEARNING")
            plt.show()

        print(l)
        L.append(l)
        # plt.plot(bias)
        # plt.show()


def create_input(x):
    k = np.zeros(10)
    k[x] = 1
    return {
        l1.id: k
    }


L2 = []


def test(plot=False):
    score = 0

    for i in range(0, 10):
        o = i % 10
        print("test iteration", i, "num:", o)
        o = create_input(o)
        x.reset_state()
        l = 0
        spikes = [[] for i in range(l2.n)]
        for j in range(0, 100):
            l += np.mean(x.tick(o)[l4.id], axis=0) / 100
            s = l2.spikes
            s[s > 0] = j
            for j, a in enumerate(s):
                if a != 0:
                    spikes[j].append(a)

        L2.append(((l - o[l1.id]) ** 2).mean(axis=0))
        final_out_pred = np.stack(l4.curr_hist, axis=0).mean(axis=0)
        print("FINAL PREDICTION:", final_out_pred)
        if np.argmax(final_out_pred) == np.argmax(o[l1.id]):
            score += 1

        if plot:
            plt.eventplot(spikes)
            plt.title(f"INPUT: {i % 10}, PREDICTION: {
                      np.argmax(final_out_pred)}")
            plt.show()

    print(f"TOTAL SCORE: {score}")


test(plot=False)
train(n=20, plot=False)
test(plot=False)
# plt.plot(L)
# plt.figure()
# plt.plot(L2)
# plt.show()
