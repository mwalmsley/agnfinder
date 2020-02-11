"""
Script version of https://colab.research.google.com/drive/1p22WgQde5ViONL8wRdONexSXL9FkZy3R#scrollTo=ylT7BWfwxRLZ
"""
import logging
import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband

from agnfinder.tf_sampling import deep_emulator

import tensorflow as tf

def main(cube_dir, hyperband_iterations, max_epochs):
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    logging.info('Found GPU at: {}'.format(device_name))

    x, y, val_x, val_y = deep_emulator.data(cube_dir)

    tuner = Hyperband(
        build_model,
        objective='mean_absolute_error',
        hyperband_iterations=hyperband_iterations,
        max_epochs=max_epochs,
        directory='results/hyperband',
        project_name='agnfinder_4layer_dropout'
    )

    early_stopping = keras.callbacks.EarlyStopping()

    tuner.search(
        x,
        y,
        callbacks=[early_stopping],
        validation_data=(val_x, val_y),
        batch_size=1024
    )

    tuner.results_summary()

    models = tuner.get_best_models(num_models=5)

    for n, model in enumerate(models):
        logging.info(f'Model {n}')
        logging.info(model.summary())

# need to change input_dim to n params and output_dim to n bands
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(
        input_dim=9,
        units=hp.Int('units1',
                    min_value=128,
                    max_value=1024,
                    step=128),
        activation='relu')
    )
    model.add(layers.Dense(
        units=hp.Int('units2',
                    min_value=128,
                    max_value=128 * 100,
                    step=128),
        # 1024,
        activation='relu')
    )
    model.add(layers.Dense(
        units=hp.Int('units3',
                    min_value=128,
                    max_value=1024,
                    step=128),
        # 128,
        activation='relu')
    )
    model.add(layers.Dense(
        units=hp.Int('units4',
                    min_value=128,
                    max_value=1024,
                    step=128),
        # 128,
        activation='relu')
    )
    # model.add(layers.Dense(
    #     units=hp.Int('units5',
    #                 min_value=128,
    #                 max_value=1024,
    #                 step=64),
    #     # 1024,
    #     activation='relu')
    # )
    model.add(layers.Dropout(hp.Float(
        'dropout',
        min_value=0.,
        max_value=0.7
    )))
    model.add(layers.Dense(8))  # default
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error'])
    return model

if __name__ == '__main__':


    tf.config.optimizer.set_jit(True)  # XLA compilation for keras model

    cube_dir = 'data/cubes/latest'
    hyperband_iterations = 2
    max_epochs = 5
    main(cube_dir, hyperband_iterations, max_epochs)
