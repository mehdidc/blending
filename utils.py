from hp_toolkit.hp import Param
import numpy as np
from skimage.io import imread

from lasagne.datasets.mnist import MNIST
from skimage.filter import threshold_otsu
from skimage.transform import resize
from sklearn.utils import shuffle
import os
W, H = 28, 28


def load_textures(w=W, h=H):
    texture = imread("1.1.07.tiff") / 255
    texture = texture.astype(np.float32)
    texture = resize(texture, (w, h)).astype(np.float32)
    return [texture]


def load_images(w=W, h=H):

    fname = "cache/mnist_{0}_{1}".format(w, h)
    if os.path.exists(fname):
        return np.load(fname)
    else:
        data = MNIST()
        data.load()

        X = data.X.reshape((data.X.shape[0], data.img_dim[0], data.img_dim[1]))
        X = shuffle(X)
        X = X[0:10000]

        X_b = np.zeros((X.shape[0], w, h), dtype=np.float32)
        for i in range(X_b.shape[0]):
                X_b[i] = resize(X[i], (w, h))
        X = X_b
        np.save(fname, X)
        return X


params_batch_optimizer = dict(
    batch_size = Param(initial=128, interval=[10, 50, 100, 128, 256, 512], type='choice'),
    learning_rate = Param(initial=10e-4, interval=[-5, -2], type='real', scale='log10'),
    momentum = Param(initial=0.5, interval=[0.5, 0.8, 0.9, 0.95, 0.99], type='choice'),
    max_epochs = Param(initial=100, interval=[100, 200], type='choice')
)
