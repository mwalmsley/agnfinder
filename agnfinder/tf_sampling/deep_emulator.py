import os
import logging
import glob

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import h5py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def tf_model():
    # note: the relu's make a huge improvement here over default (sigmoid?)

    # previous default
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(256, input_dim=7, activation='relu'),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dropout(0.08),
    #     tf.keras.layers.Dense(12)
    #     ])

    # current best from hyperband w/ 1m cube, 15 epochs
    # TODO found before redshift was introduced, could update
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(192, input_dim=8, activation='relu'),
        tf.keras.layers.Dense(448, activation='relu'),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dense(576, activation='relu'),
        tf.keras.layers.Dropout(0.004),
        tf.keras.layers.Dense(12)
        ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error'])
    return model

def data(cube_dir):  # e.g. data/cubes/latest
    # need to be able to load all cubes into memory at once (though only once, thanks to concatenation instead of loading all and stacking)
    cube_locs = get_cube_locs(cube_dir)
    theta = np.ones((0, 8))
    normalised_photometry = np.ones((0, 12))
    for cube_loc in cube_locs:
        cube_theta, cube_normalised_photometry = load_cube(cube_loc)
        theta = np.concatenate([theta, cube_theta], axis=0)
        normalised_photometry = np.concatenate([normalised_photometry, cube_normalised_photometry], axis=0)
    logging.info('Loaded {} theta, {} photometry'.format(theta.shape, normalised_photometry.shape))
    # shuffles here, which is crucial
    x_train, x_test, y_train, y_test = train_test_split(theta, normalised_photometry, random_state=1, test_size=0.02)
    return x_train, y_train, x_test, y_test


def train_manual(model, train_features, train_labels, test_features, test_labels):
    early_stopping = tf.keras.callbacks.EarlyStopping()
    model.fit(train_features, train_labels, epochs=15, validation_data=(test_features, test_labels), callbacks=[early_stopping])
    return model


def normalise_photometry(photometry):
    return -1 * np.log10(photometry)


def denormalise_photometry(normed_photometry):
    return 10 ** (-1 * normed_photometry)


def train_boosted_trees(cube_dir):
    x_train, y_train, x_test, y_test = data(cube_dir)
    clf = GradientBoostingRegressor().fit(x_train, y_train)
    print(mean_squared_error(y_test, clf.predict(x_test))) 


def get_trained_keras_emulator(emulator, checkpoint_dir, new=False, cube_dir=None):
    checkpoint_loc = os.path.join(checkpoint_dir, 'model')
    if new:
        logging.info('Training new emulator')
        assert cube_dir is not None  # need cubes to make a new emulator!
        emulator = train_manual(emulator, *data(cube_dir))
        emulator.save_weights(checkpoint_loc)
    else:
        logging.info('Loading previous emulator from {}'.format(checkpoint_loc))
        emulator.load_weights(checkpoint_loc)  # modifies inplace
    return emulator


def get_cube_locs(cube_dir):
    cube_locs = glob.glob(os.path.join(cube_dir, 'photometry_simulation_*.hdf5'))  # only place cubes with matching n, z!
    # TODO check matching n, z?
    assert cube_locs
    logging.warning('Using cube locs: {}'.format(cube_locs))
    return cube_locs


def cubes_to_tfrecords(cube_dir, tfrecord_dir):
    cube_locs = get_cube_locs(cube_dir)
    for cube_loc in cube_locs:
        logging.debug('Loading and saving cube {}'.format(cube_loc))
        theta, norm_photometry = load_cube(cube_loc)
        ds = tf.data.Dataset.from_tensor_slices((theta, norm_photometry))
        serialized_ds = ds.map(tf_serialize_example)
        tfrecord_loc = cube_loc.rstrip('.hdf5') + '.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(tfrecord_loc)
        writer.write(serialized_ds)
    logging.info('Saved cubes in {} as tfrecords to {}'.format(cube_dir, tfrecord_dir))

    # now write as fixed train/test records
def tfrecords_to_train_test(tfrecord_dir):
    tfrecord_locs = glob.glob(os.path.join(tfrecord_dir, 'photometry_simulation_*.tfrecord'))
    ds = tf.data.load_interleaved()

def parse_example():
    pass

def _floats_feature(x):  # should be a 1D array or list
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(x)))


def serialize_example(theta, norm_photometry):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'theta': _floats_feature(theta),
        'norm_photometry': _floats_feature(norm_photometry),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(theta, norm_photometry):
    tf_string = tf.py_function(
        serialize_example,
        (theta, norm_photometry),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar


def load_cube(loc):
    with h5py.File(loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        # very important to remember to also do this to real data!
        norm_photometry = normalise_photometry(f['samples']['simulated_y'][...])
    # theta as features, normalised_photometry as labels
    return theta, norm_photometry



if __name__ == '__main__':
    """
    You can run this (with no args) to train a new emulator (which will be saved as below)

    To find better model hyperparams, see Google Colab
    """
    # tf.config.optimizer.set_jit(True)  # may break everything
    tf.enable_eager_execution()

    # train_boosted_trees()

    # create_new_emulator
    cube_dir = 'data/cubes/latest'
    checkpoint_dir = 'results/checkpoints/redshift_test'  # last element is checkpoint name
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    cubes_to_tfrecords(cube_dir, cube_dir)

    # model = tf_model()
    # trained_clf = get_trained_keras_emulator(model, checkpoint_dir, new=True, cube_dir=cube_dir)
