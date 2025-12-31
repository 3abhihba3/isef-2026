import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Reservoir:
    min_rate = 0.1
    max_rate = 0.5

    def __init__(self, n, inhib=0.5):
        # not learned
        self.potentials = np.zeros(n, dtype=np.float16)
        self.spike_sizes = np.ones(n, dtype=np.int8)
        self.spike_sizes *= -2 * np.random.binomial(1, inhib, n) + 1
        self.spikes = np.zeros(n, dtype=bool)

        # learned, homeostatic
        self.thresholds = np.ones(n, dtype=np.float16) * 0.4
        self.decays = np.ones(n, dtype=np.float16) * 0.6

        # learned, shallow
        self.biases = np.ones(n, dtype=np.float16) * 0.16

        # learned, deep
        self.weights = np.random.random_sample(
            (n, n)).astype(np.float16) * (2/n)
        self.weights *= np.random.binomial(1, 0.3, size=self.weights.shape)

        # may be replaced by threshold updates
        self.offsets = np.ones(n, dtype=np.float16) * 0.1

    def tick(self, curr_in):
        spike_locations = (self.potentials > self.thresholds)
        spikes = spike_locations * self.spike_sizes
        self.potentials[spike_locations] = 0

        curr = self.weights @ spikes
        self.potentials = self.potentials * self.decays + \
            curr + curr_in + self.biases  # + self.offsets

        # learn parameters
        return spike_locations

    def _homeostatic_updates(self):
        pass

    def _shallow_updates(self):
        pass

    def _deep_updates(self):
        pass

# random stuff under here


def quadratic_kernel(k):
    n = np.arange(-(k//2), k//2 + 1)
    w = 1 - (n / (k//2))**2
    # w[w < 0] = 0
    w += w.min()
    return w / w.sum()


def input_current(t, N):
    return np.max(np.sin(t / 20), 0) * np.ones(N, dtype=np.float16) if t < 500 else np.zeros(N)


def main(N):
    T = 1000
    k = 39
    kernel = quadratic_kernel(k)
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%.9g" % x))

    spikes = np.empty((T, N), dtype=np.int8)
    activation_mean = np.empty((max(T, k) - min(T, k) + 1, 3))
    input_hist = np.empty((T, N))
    mask = np.random.binomial(1, 0.2, N)

    res = Reservoir(N)
    print(res.weights)
    for i in range(T):
        input_hist[i] = input_current(i, N) * mask
        y = res.tick(input_hist[i])
        spikes[i] = y
        # print(res.potentials)
        #
    for i in range(3):
        # print(spikes[:, i])
        activation_mean[:, i] = np.convolve(
            spikes[:, i + 4], kernel, mode='valid')

    # print("N:", N, "SPIKES:", np.mean(spikes))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = activation_mean.T
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    line, = ax.plot([], [], [], lw=1)
    point, = ax.plot([], [], [], 'o')

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties(z[frame])
        return line, point

    anim = FuncAnimation(fig, update, frames=len(x), interval=50)
    plt.figure()
    plt.plot(np.linalg.norm(input_hist, axis=1))
    plt.show()


main(100)
