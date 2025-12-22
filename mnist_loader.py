import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms


def load_mnist_numpy(
    root="./data",
    train=True,
    max_samples=None,
    shuffle=True,
    seed=0,
):
    """
    Returns:
        images: (N, 28, 28) float32 in [0, 1)
        labels: (N,) int64
    """
    transform = transforms.ToTensor(
    )  # converts to [0,1] float32, shape (1,28,28)

    dataset = MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    N = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(indices)
    indices = indices[:N]

    images = np.empty((N, 28, 28), dtype=np.float32)
    labels = np.empty(N, dtype=np.int64)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        images[i] = img.numpy().squeeze(0)   # (28,28)
        labels[i] = label

    return images, labels


if __name__ == "__main__":
    X, y = load_mnist_numpy(train=True, max_samples=1000)

    print(X.shape, X.dtype, X.min(), X.max())
    print(y.shape, y.dtype)
