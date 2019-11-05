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


def tf_model():
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

# because hyperas is weird, this isn't allowed any arguments - not even via closure! TODO
def data():
    loc = 'data/photometry_simulation_1000000.hdf5'
    # loc = 'data/photometry_simulation_100000.hdf5'
    logging.warning('Using data loc {}'.format(loc))
    assert os.path.isfile(loc)
    with h5py.File(loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        # hacky extra normalisation here, not great TODO
        simulated_y = -1 * np.log10(f['samples']['simulated_y'][...])
    features = theta
    labels = simulated_y
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    return x_train, y_train, x_test, y_test

def train_manual(model, train_features, train_labels, test_features, test_labels):
    early_stopping = keras.callbacks.EarlyStopping()
    model.fit(train_features, train_labels, epochs=3, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
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


def train_boosted_trees():
    x_train, y_train, x_test, y_test = data()
    clf = GradientBoostingRegressor().fit(x_train, y_train)
    print(mean_squared_error(y_test, clf.predict(x_test))) 


def get_trained_keras_emulator(emulator, checkpoint_loc, new=False):
    if new:
        emulator = train_manual(emulator, *data())
        emulator.save_weights(checkpoint_loc)
    else:
        emulator.load_weights(checkpoint_loc)  # modifies inplace
    return emulator


def find_best_model(max_evals=40):
    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=max_evals,
                                        trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)


if __name__ == '__main__':

    tf.enable_eager_execution()

    # train_boosted_trees()

    # find_best_model()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(test_features, test_labels))

    # create_new_emulator
    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'  # last element is checkpoint name
    model = tf_model()
    trained_clf = get_trained_keras_emulator(model, checkpoint_loc, new=True)
