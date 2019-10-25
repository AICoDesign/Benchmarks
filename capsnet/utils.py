import numpy as np
from matplotlib import pyplot as plt
import csv
import math
from collections import defaultdict, Counter
import warnings
from zipfile import ZipFile
# from urllib import urlretrieve
from six.moves.urllib.request import urlretrieve
import six
from tempfile import mktemp
import fnmatch
import pip
import scipy
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import random

import os

# Keras imports
import tensorflow as tf
from keras import initializers, callbacks, optimizers, layers, models, backend as K
from keras.callbacks import Callback

K.set_image_data_format('channels_last')


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

class GetBest(Callback):
    """Get the best model at the end of training.
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


"""Utils"""


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        tf.set_random_seed(seed)


def install(package):
    pip.main(['install', package])


def dl_tumor_data(path):
    print("Downloading tumor data")
    os.makedirs(path)
    url = "https://ndownloader.figshare.com/articles/1512427/versions/5"
    filename = mktemp('.zip')
    destDir = 'data'
    urlretrieve(url, filename)
    file = ZipFile(filename)
    file.extractall(destDir)
    file.close()

    for root, dirnames, filenames in os.walk('data'):
        for filename in fnmatch.filter(filenames, '*.zip'):
            fn = os.path.join(root, filename)
            file = ZipFile(fn)
            file.extractall(path)
            file.close()


def load_tumor():
    try:
        import h5py
    except ImportError:
        print("Installing h5py package to read matlab files")
        install('h5py')

    print("Loading tumor sets")

    path = 'data/tumor_data'
    if not os.path.exists(path):
        dl_tumor_data(path)

    p_imgs = defaultdict(list)
    p_type = {}
    m = 0
    for file in os.listdir(path):
        f = h5py.File(os.path.join(path, file), 'r')
        label = int(f.get('cjdata/label')[0][0])
        p = f.get('cjdata/PID')
        pid = str(''.join([six.unichr(x[0]) for x in p]))
        img = np.array(f.get('cjdata/image'))
        #img = scipy.misc.imresize(img, (64, 64))
        img = img.astype('float32') / 12728
        img = resize(img, (64, 64))
        #img *= (255.0 / img.max())
        img = np.round(img)
        l = [0, 0, 0]
        l[label - 1] = 1
        p_imgs[pid].append(img)
        p_type[pid] = l
    print(m)
    X = list(p_type.keys())
    y = list(p_type.values())

    tts_split = train_test_split(
        X, y, range(len(y)), test_size=0.3, random_state=0, stratify=y)

    pX_train, pX_test, py_train, py_test, train_idx, test_idx = tts_split

    X_train, X_test, y_train, y_test, train_recon, test_recon = [], [], [], [], [], []

    for id in pX_train:
        x = p_imgs[id]
        y_train.extend([p_type[id]] * len(x))
        x = np.stack(x).reshape(len(x), 64, 64, 1)#.astype('float64') / 255
        X_train.extend(x)

    for id in pX_test:
        x = p_imgs[id]
        rge = len(X_test) + len(x)
        if len(x) > 1:
            rge -= 1

        p_recon = ([len(X_test), rge], p_type[id])
        test_recon.append(p_recon)
        y_test.extend([p_type[id]] * len(x))
        x = np.stack(x).reshape(len(x), 64, 64, 1)#.astype('float64') / 255
        X_test.extend(x)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test[:-300], y_train, y_test[:-300], X_test[-300:], y_test[-300:], test_recon

if __name__=="__main__":
    plot_log('result/log.csv')


