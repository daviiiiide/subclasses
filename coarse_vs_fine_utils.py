import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input
from sklearn.model_selection import train_test_split
import keras.backend as K

import urllib.request
import gzip
import numpy as np
from io import BytesIO

def load_kmnist():
    """Download and load KMNIST dataset directly into memory"""
    
    base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    def download_and_parse(url, is_image=True, n_samples=60000):
        print(f"Downloading {url.split('/')[-1]}...")
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=BytesIO(response.read())) as f:
                if is_image:
                    f.read(16)  # skip header
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    return data.reshape(n_samples, 28, 28)
                else:
                    f.read(8) 
                    return np.frombuffer(f.read(), dtype=np.uint8)
    
    X_train = download_and_parse(base_url + files["train_images"], True, 60000)
    y_train = download_and_parse(base_url + files["train_labels"], False)
    X_test = download_and_parse(base_url + files["test_images"], True, 10000)
    y_test = download_and_parse(base_url + files["test_labels"], False)
    
    print(f"KMNIST loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def class_sizer(X_train, Y_train, data_per_class, n_classes_fine):
    for i in range(n_classes_fine):
        data_matrix_partial = X_train[Y_train == i][:data_per_class][:]
        label_matrix_partial = np.empty(data_per_class)
        if i == 0:
            data_matrix = data_matrix_partial
            label_matrix_partial[:] = i
            label_matrix = label_matrix_partial.copy()
        else:
            data_matrix = np.vstack((data_matrix, data_matrix_partial))
            label_matrix_partial[:] = i
            label_matrix = np.hstack((label_matrix, label_matrix_partial))
    return data_matrix, label_matrix


def coarser(Y, conf):
    Y_coarse = np.zeros((Y.shape[0])) 
    for i in range(Y.shape[0]):
        Y_coarse[i] = conf[int(Y[i])]
    return Y_coarse


def weights_creator(conf):
    a_tot = np.array([conf])
    w_tot = np.empty(shape=(a_tot.shape[0], conf.shape[0], 2))
    for i in range(w_tot.shape[0]):
        for j in range(w_tot.shape[1]):
            if a_tot[i, j] == 0:
                w_tot[i, j, 0] = 1
                w_tot[i, j, 1] = 0
            if a_tot[i, j] == 1:
                w_tot[i, j, 0] = 0
                w_tot[i, j, 1] = 1
    return w_tot


def class_exchanger(y, a, b):
    y_new = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        y_new[i] = y[i]
        if y[i] == a:
            y_new[i] = b
        if y[i] == b:
            y_new[i] = a
    return y_new


class LinearLearningRateScheduler(Callback):
    def __init__(self, initial_lr, decay, limit, verbose=0):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay = decay
        self.limit = limit
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = max(self.initial_lr - epoch * self.decay, self.limit)
        optimizer = self.model.optimizer
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer.learning_rate.assign(new_lr)
        else:
            raise TypeError(f"Expected optimizer object, but got {type(optimizer)}")
        
        if self.verbose > 0:
            print(f"Epoch {epoch + 1}: Learning rate is set to {new_lr:.6f}")


def run_experiment(lay1, size, bs_coarse, decay_rate, ep_of_diminish, lr1, ep_coarse, exp, 
                   X_fix, y_train, X_test, y_test, conf, n_classes_fine, val_size, 
                   verbose, lr_version, dataset):
    keras.backend.clear_session()
    
    (x, y) = class_sizer(X_fix, y_train, size, n_classes_fine)
    
    X_train_fine = np.copy(x)
    Y_train_fine = np.copy(y)
    X_test_fine = np.copy(X_test)
    Y_test_fine = np.copy(y_test)
    
    X_test_fine, X_val_fine, Y_test_fine, Y_val_fine = train_test_split(
        X_test_fine, Y_test_fine, test_size=val_size, stratify=Y_test_fine)    
    
    X_train_coarse = np.copy(X_train_fine)
    X_test_coarse = np.copy(X_test_fine)
    X_val_coarse = np.copy(X_val_fine)
    
    Y_train_coarse = np.copy(coarser(np.copy(Y_train_fine), conf))
    Y_test_coarse = np.copy(coarser(np.copy(Y_test_fine), conf))
    Y_val_coarse = np.copy(coarser(np.copy(Y_val_fine), conf))
    
    shape = X_train_fine[0].shape
    
    if lr_version == 'constant':
        lr_schedule = lr1
        momentum = 0.5
    elif lr_version == 'customized':
        if dataset == 'cifar10':
            lr_schedule = 0.001
            lr_custom = LinearLearningRateScheduler(initial_lr=0.001, decay=round(lr_schedule/130,2), limit=0.0001, verbose=0)
            momentum = 0.0
        else:
            lr_schedule = 0.012
            lr_custom = LinearLearningRateScheduler(initial_lr=0.012, decay=0.00005, limit=0.001, verbose=0)
            momentum = 0.5
    elif lr_version == 'decay':
        initial_learning_rate = lr1
        decay_steps = int((size * 8 * ep_of_diminish) / (bs_coarse))
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
        momentum = 0.5
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0005, patience=60)
    
    Y_train_coarse_cat = keras.utils.to_categorical(Y_train_coarse, 2)
    Y_val_coarse_cat = keras.utils.to_categorical(Y_val_coarse, 2)
    Y_test_coarse_cat = keras.utils.to_categorical(Y_test_coarse, 2)
    
    model_coarse = Sequential([
        Input(shape=shape),
        Flatten(),
        Dense(lay1, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        Dense(2, activation='softmax')
    ])
    
    model_coarse.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum),
        metrics=['accuracy']
    )
    
    if lr_version == 'customized':
        callbacks = [callback, lr_custom]
    else:
        callbacks = [callback]
    
    history_coarse = model_coarse.fit(
        X_train_coarse,
        Y_train_coarse_cat,
        epochs=ep_coarse,
        batch_size=bs_coarse,
        validation_data=(X_val_coarse, Y_val_coarse_cat),
        verbose=verbose,
        callbacks=callbacks,
        shuffle=True
    )
    
    test_loss_coarse, acc_coarse = model_coarse.evaluate(X_test_coarse, Y_test_coarse_cat, verbose=0)
    train_loss_coarse, acc_coarse_train = model_coarse.evaluate(X_train_coarse, Y_train_coarse_cat, verbose=0)
    stop_ep_coarse = callback.stopped_epoch
    
    keras.backend.clear_session()
    
    Y_train_fine_cat = keras.utils.to_categorical(Y_train_fine, n_classes_fine)
    Y_val_fine_cat = keras.utils.to_categorical(Y_val_fine, n_classes_fine)
    Y_test_fine_cat = keras.utils.to_categorical(Y_test_fine, n_classes_fine)
    
    model_fine = Sequential([
        Input(shape=shape),
        Flatten(),
        Dense(lay1, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        Dense(n_classes_fine, activation='softmax')
    ])
    
    model_fine.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum),
        metrics=['accuracy']
    )
    
    history_fine = model_fine.fit(
        X_train_fine,
        Y_train_fine_cat,
        epochs=ep_coarse,
        batch_size=bs_coarse,
        validation_data=(X_val_fine, Y_val_fine_cat),
        verbose=verbose,
        callbacks=callbacks,
        shuffle=True
    )
    
    test_loss_fine_nosum, acc_fine_ns = model_fine.evaluate(X_test_fine, Y_test_fine_cat, verbose=0)
    train_loss_fine_nosum, acc_fine_train_ns = model_fine.evaluate(X_train_fine, Y_train_fine_cat, verbose=0)
    
    weights = weights_creator(conf)
    model_fine.add(Dense(2, use_bias=False, trainable=False))
    w = np.reshape(weights, np.shape(model_fine.layers[-1].get_weights()))
    model_fine.layers[-1].set_weights(w)
    
    params = model_coarse.count_params()
    test_loss_fine, acc_fine = model_fine.evaluate(X_test_coarse, Y_test_coarse_cat, verbose=0)
    train_loss_fine, acc_fine_train = model_fine.evaluate(X_train_coarse, Y_train_coarse_cat, verbose=0)
    stop_ep_fine = callback.stopped_epoch
    
    if verbose > 0:
        print(f'Completed experiment: Layer={lay1}, Size={size}, Exp={exp+1}, LR={lr1}, BS={bs_coarse}')
        print(f'Coarse accuracy: {acc_coarse}, Fine accuracy: {acc_fine}')
    
    return {
        'Accuracy_coarse': acc_coarse,
        'Accuracy_fine': acc_fine,
        'Accuracy_training_coarse': acc_coarse_train,
        'Accuracy_training_fine': acc_fine_train,
        'Test loss coarse': test_loss_coarse,
        'Train loss coarse': train_loss_coarse,
        'Test loss fine': test_loss_fine,
        'Train loss fine': train_loss_fine,
        'Test loss fine nosum': test_loss_fine_nosum,
        'Train loss fine nosum': train_loss_fine_nosum,
        'Size': size * n_classes_fine,
        'Single_Class_Size': size,
        'Ratio': round((params / (size * n_classes_fine)), 3),
        'Experiment': exp + 1,
        'Nodes': lay1,
        'Batch Size': bs_coarse,
        'Epochs': ep_coarse,
        'Epoch of stop Coarse': stop_ep_coarse,
        'Epoch of stop Fine': stop_ep_fine,
        'Diminish epochs': ep_of_diminish,
        'Decay rate': decay_rate,
        'Learning Rate': lr1
    }