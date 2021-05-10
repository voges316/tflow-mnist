#!/usr/bin/env python3

"""Fashion-MNIST dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file


def load_fashion_data(download=False):
    """Loads the Fashion-MNIST dataset.

  This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
  along with a test set of 10,000 images. This dataset can be used as
  a drop-in replacement for MNIST. The class labels are:

  | Label | Description |
  |:-----:|-------------|
  |   0   | T-shirt/top |
  |   1   | Trouser     |
  |   2   | Pullover    |
  |   3   | Dress       |
  |   4   | Coat        |
  |   5   | Sandal      |
  |   6   | Shirt       |
  |   7   | Sneaker     |
  |   8   | Bag         |
  |   9   | Ankle boot  |

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: uint8 arrays of grayscale image data with shape
        (num_samples, 28, 28).

      **y_train, y_test**: uint8 arrays of labels (integers in range 0-9)
        with shape (num_samples,).

  License:
      The copyright for Fashion-MNIST is held by Zalando SE.
      Fashion-MNIST is licensed under the [MIT license](
      https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

    """
    dirname = os.path.join('data', 'fashion')
    base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        if download:
            # Keras cache dir defaults to the default directory ~/.keras/.
            paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))
        else:
            paths.append(os.path.join(dirname, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
