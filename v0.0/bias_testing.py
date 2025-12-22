import numpy as np
import matplotlib.pyplot as plt

target = np.array([0, 0.4, 0.3, 0.1])
bias = np.zeros(4)

k = []
l = []
e = []
r = []

_input = np.zeros(4)

for i in range(1000):
    _input += np.random.random_sample(4) * 0.1 - 0.05
    total_in = _input + bias
    bias += (target - total_in) / 20
    # _input += bias / 20
    e.append(np.mean((bias - target) ** 2))
    l.append(np.linalg.norm(bias))
    k.append(total_in)
    r.append(np.linalg.norm(total_in))

plt.figure()
[plt.plot(i, color='blue', alpha=_/len(k)) for _, i in enumerate(k)]
plt.figure()
plt.plot(l, color='red')
plt.plot(e, color='blue')
plt.plot(r, color='green')
plt.show()
