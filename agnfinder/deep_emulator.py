import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import h5py

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def network():
    model = Sequential()
    model.add(Dense(512, input_dim=7))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Dropout(0.6))
    model.add(Dense(12))
    model.add(Activation('relu'))
    return model

def tf_model():
    # inputs = tf.keras.Input(shape=(7,))
    # x0 = tf.keras.layers.Dense(512)(inputs)
    # x1 = tf.keras.layers.Dense(256)(x0)
    # x2 = tf.keras.layers.Dense(512)(x1)
    # x3 = tf.keras.layers.Dropout(0.6)(x2)
    # outputs = tf.keras.layers.Dense(12)(x3)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(
    #     optimizer='adam',
    #     loss='mean_squared_error',
    #     metrics=['mean_absolute_error'])
    # return model
    
    # note: the relu's make a huge improvement here over default (sigmoid?)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256,input_dim=7, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.08),
        tf.keras.layers.Dense(12)
        ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error'])
    return model


def data():
    
    local_loc = '/media/mike/internal/agnfinder/photometry_simulation_1000000.hdf5'
    cluster_loc = 'data/photometry_simulation_1000000.hdf5'
    if os.path.exists(local_loc):
        data_loc = local_loc
    else:
        data_loc = cluster_loc

    logging.warning('Using data loc {}'.format(data_loc))
    assert os.path.isfile(data_loc)
    with h5py.File(data_loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        # hacky extra normalisation here, not great TODO
        simulated_y = -1 * np.log10(f['samples']['simulated_y'][...])
    features = theta
    labels = simulated_y

    # x_train, x_test, y_train, y_test = train_test_split(features, labels[:, 4].reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(features, labels)
    return x_train, y_train, x_test, y_test

def train_manual(model, train_features, train_labels, test_features, test_labels):



    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_features, train_labels)
    # )
    # dataset = dataset.shuffle(10000).batch(64)
    # optimizer = tf.train.AdamOptimizer()

    # @tf.function()
    # def train():
    #     # with tf.variable_scope('training', reuse=tf.AUTO_REUSE):
    #     for (batch, (features, labels)) in enumerate(dataset):
    #     # if batch % 100 == 0:
    #         with tf.GradientTape() as tape:
    #             predictions = clf(features, training=True)
    #             loss_value = tf.losses.mean_squared_error(labels, predictions)
    #             grads = tape.gradient(loss_value, clf.trainable_variables)
    #             optimizer.apply_gradients(zip(grads, clf.trainable_variables),
    #                                     global_step=tf.train.get_or_create_global_step())


    model.fit(train_features, train_labels, epochs=3, batch_size=64, validation_split=0.2)

    # test_loss, test_abs_error = model.evaluate(test_features, test_labels)
    # print(test_loss, test_abs_error)

    return model


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([256, 512, 1024])}}, input_dim=7))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(12))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'],
                    optimizer='adam')

    result = model.fit(x_train, y_train,
                batch_size={{choice([64, 128])}},
                epochs=40,
                verbose=2,
                validation_split=0.2)
    #get the highest validation accuracy of the training epochs TODO bad...?
    n_best = 5
    best_val_losses = np.sort(result.history['val_loss'])[:n_best].mean()
    print('Mean best loss (of {}) of epoch:'.format(n_best), best_val_losses)
    return {'loss': best_val_losses, 'status': STATUS_OK, 'model': model}


def boosted_trees():
    x_train, y_train, x_test, y_test = data()
    clf = GradientBoostingRegressor().fit(x_train, y_train)
    print(mean_squared_error(y_test, clf.predict(x_test))) 

if __name__ == '__main__':

    

    # boosted_trees()
    # exit()

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=40,
                                          trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(test_features, test_labels))

    # checkpoint_dir = 'results/trained_deep_emulator/checkpoint'

    # tf.enable_eager_execution()
    # model = tf_model()
    # trained_clf = train_manual(model, *data())

    # print('Training complete - saving')
    # checkpointer = tf.train.Checkpoint(model=trained_clf)
    # checkpointer.save(checkpoint_dir)
