import nengo
import numpy as np
import matplotlib.pyplot as plt

with nengo.Network() as net:
    sin_input = nengo.Node(output=np.sin)
    # A population of 100 neurons representing a sine wave
    sin_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(sin_input, sin_ens)
    # A population of 100 neurons representing the square of the sine wave
    sin_squared = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(sin_ens, sin_squared, function=np.square)
    # View the decoded output of sin_squared
    squared_probe = nengo.Probe(sin_ens, synapse=0.01, attr='decoded_output')

with nengo.Simulator(net) as sim:
    sim.run(5.0)
plt.plot(sim.trange(), sim.data[squared_probe])
plt.show()
