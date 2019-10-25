# Keras imports
import tensorflow as tf
from keras import initializers, callbacks, optimizers, layers, models, backend as K
from keras.callbacks import Callback

K.set_image_data_format('channels_last')

# Other imports
import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import argparse
from collections import defaultdict
import warnings
from zipfile import ZipFile
# from urllib import urlretrieve
from six.moves.urllib.request import urlretrieve
import six
from tempfile import mktemp
import fnmatch
import pip
from collections import Counter
import random
from utils import combine_images
from PIL import Image


"""
Capsnet Keras Implementation Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """

    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """

    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides,
                           padding=padding,
                           name='primarycap_conv2d')(inputs)

    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(256, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(512, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


"""Custom callback for saving the best model during training"""


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

    path = 'data/tumor'
    if not os.path.exists(path):
        dl_tumor_data(path)

    p_imgs = defaultdict(list)
    p_type = {}
    for file in os.listdir(path):
        f = h5py.File(os.path.join(path, file), 'r')
        label = int(f.get('cjdata/label')[0][0])
        p = f.get('cjdata/PID')
        pid = str(''.join([six.unichr(x[0]) for x in p]))
        img = np.array(f.get('cjdata/image'))
        img = scipy.misc.imresize(img, (64, 64))
        img = (img.astype('float32') / 65535)
        img *= (255.0 / img.max())
        img = np.round(img)
        l = [0, 0, 0]
        l[label - 1] = 1
        p_imgs[pid].append(img)
        p_type[pid] = l

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


def params():
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on brain tumor classification.")
    parser.add_argument('--epochs', default=30, type=int, help="Number of epochs to train for")
    parser.add_argument('--verb', default=1, type=int, help="Whether to print train progress, 0 for no output")
    parser.add_argument('--sub', default=0, type=int, help="Number of training samples to use, max=2100")
    parser.add_argument('--batch_size', default=100, type=int, help="Number of images per batch")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should be > 0")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights")
    parser.add_argument('--train', action='store_true',
                        help="Train/retrain the loaded model?")
    parser.add_argument('--type', default=None, type=int,
                        help="Tumor type to manipulate")
    parser.add_argument('-seed', default=42, type=int,
                        help="Seed to use")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def get_models(args, shape):
    model, eval_model, manipulate_model = CapsNet(input_shape=shape, n_class=3, routings=args.routings)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    # Eval model does not use labels to mask capsnet output
    # instead it uses the capsule with greatest length (probability)
    '''eval_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                       loss=[margin_loss, 'mse'],
                       loss_weights=[1., args.lam_recon],
                       metrics={'capsnet': 'accuracy'})

    manipulate_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                       loss=[margin_loss, 'mse'],
                       loss_weights=[1., args.lam_recon],
                       metrics={'capsnet': 'accuracy'})'''

    if args.weights:
        model.load_weights(args.save_dir + '/weights/' + args.weights)
        eval_model.load_weights(args.save_dir + '/weights/' + args.weights)
        manipulate_model.load_weights(args.save_dir + '/weights/' + args.weights)

    return model, eval_model, manipulate_model


def train_model(model, X_train, X_test, y_train, y_test, args):
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    gb = GetBest(monitor='val_capsnet_acc', verbose=0, mode='max')
    lr_red = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    model.fit([X_train, y_train], [y_train, X_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[X_test, y_test], [y_test, X_test]], callbacks=[lr_decay, gb, lr_red, checkpoint], verbose=args.verb)


def test_model(eval_model, X_test, y_test, X_hold, y_hold, test_recon, args):
    y_pred, _ = eval_model.predict(X_test, batch_size=args.batch_size)
    y_pred = np.argmax(y_pred, 1)
    print('Test acc:', np.sum(y_pred == np.argmax(y_test, 1)) / float(y_test.shape[0]))

    y_hold_pred, _ = eval_model.predict(X_hold, batch_size=args.batch_size)
    y_hold_pred = np.argmax(y_hold_pred, 1)
    print('Hold acc:', np.sum(y_hold_pred == np.argmax(y_hold, 1)) / float(y_hold.shape[0]))

    y_pred = np.concatenate([y_pred, y_hold_pred])
    tc = 0
    for p in test_recon:
        rge, label = p
        if Counter(y_pred[rge[0]: rge[1]]).most_common(1)[0][0] == np.argmax(label):
            tc += 1

            # print('Majority test acc:', tc / float(len(test_recon)))


def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.type
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 3, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.type)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.type))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def main():
    args = params()
    set_seed(args.seed)

    X_train, X_test, y_train, y_test, X_hold, y_hold, test_recon = load_tumor()
    model, eval_model, manipulate_model = get_models(args, X_train.shape[1:])

    if args.sub > 0:
        X_train = X_train[:args.sub]
        y_train = y_train[:args.sub]

    if args.train:
        print("Training on %d images, testing on %d images with %d holdout" % (len(y_train), len(y_test), len(y_hold)))
        train_model(model, X_train, X_test, y_train, y_test, args)
    test_model(eval_model, X_test, y_test, X_hold, y_hold, test_recon, args)
    if args.type in [0,1,2]:
        manipulate_latent(manipulate_model, (np.concatenate([X_test, X_hold]), np.concatenate([y_test, y_hold])), args)


if __name__ == '__main__':
    main()
