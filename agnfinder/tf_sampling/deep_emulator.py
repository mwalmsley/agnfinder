import os
import logging
import glob
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tqdm as tqdm
import h5py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def tf_model(input_dim=9, output_dim=8):
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

    # hyperband w/ 1m cube, 15 epochs
    # found before redshift was introduced
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(192, input_dim=input_dim, activation='relu'),
    #     tf.keras.layers.Dense(448, activation='relu'),
    #     tf.keras.layers.Dense(192, activation='relu'),
    #     tf.keras.layers.Dense(576, activation='relu'),
    #     tf.keras.layers.Dropout(0.004),
    #     tf.keras.layers.Dense(output_dim)
    #     ])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(192, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(640, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dense(832, activation='relu'),
        tf.keras.layers.Dropout(0.014),
        tf.keras.layers.Dense(output_dim)
        ])



    model.compile(
        optimizer='adam',
        loss='val_mean_absolute_error',
        metrics=['val_mean_squared_error'])
    return model


def data(cube_dir, photometry_dim=8, theta_dim=9):  # e.g. data/cubes/latest
    # need to be able to load all cubes into memory at once (though only once, thanks to concatenation instead of loading all and stacking)
    cube_locs = get_cube_locs(cube_dir)
    theta = np.ones((0, theta_dim))
    normalised_photometry = np.ones((0, photometry_dim))
    for cube_loc in cube_locs:
        cube_theta, cube_normalised_photometry = load_cube(cube_loc)
        theta = np.concatenate([theta, cube_theta], axis=0)
        normalised_photometry = np.concatenate([normalised_photometry, cube_normalised_photometry], axis=0)
    logging.info('Loaded {} theta, {} photometry'.format(theta.shape, normalised_photometry.shape))
    # shuffles here, which is crucial
    x_train, x_test, y_train, y_test = train_test_split(theta, normalised_photometry, random_state=1, test_size=0.1)
    return x_train, y_train, x_test, y_test


# equivalent to the above, but for out-of-memory size data
def data_from_tfrecords(tfrecord_dir, batch_size=512):
    train_locs = glob.glob(os.path.join(cube_dir, 'train_*.tfrecord'))
    test_locs = glob.glob(os.path.join(cube_dir, 'test_*.tfrecord'))
    assert train_locs
    assert test_locs
    logging.info('Train locs: {}'.format(train_locs))
    logging.info('Test locs: {}'.format(test_locs))
    train_ds = load_from_tfrecords(train_locs, batch_size=batch_size)
    test_ds = load_from_tfrecords(test_locs, batch_size=batch_size)
    return train_ds, test_ds


def load_from_tfrecords(tfrecord_locs, batch_size, shuffle_buffer=10000):
    assert isinstance(tfrecord_locs, list)
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_locs) 
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=len(tfrecord_locs),
        block_length=1)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(parse_function)
    ds = ds.batch(batch_size)
    return ds


def train_on_cubes(model, train_features, train_labels, test_features, test_labels):
    early_stopping = tf.keras.callbacks.EarlyStopping(restore_best_weights=True)
    if len(tf.config.list_physical_devices('GPU')) > 0:
        batch_size = 1024  # crank it up to efficiently use GPU
        logging.info(f'GPU found - using batch size {batch_size}')
    else:
        batch_size = 128
    model.fit(train_features, train_labels, epochs=15, batch_size=batch_size, validation_data=(test_features, test_labels), callbacks=[early_stopping])
    return model


def train_on_datasets(model, train_ds, test_ds):
    early_stopping = tf.keras.callbacks.EarlyStopping(restore_best_weights=True)
    model.fit(train_ds, epochs=15, validation_data=test_ds, callbacks=[early_stopping])
    return model


def normalise_photometry(photometry):
    return -1 * np.log10(photometry)


def denormalise_photometry(normed_photometry):
    return 10 ** (-1 * normed_photometry)


def train_boosted_trees(cube_dir):
    x_train, y_train, x_test, y_test = data(cube_dir)
    clf = GradientBoostingRegressor().fit(x_train, y_train)
    print(mean_squared_error(y_test, clf.predict(x_test))) 


def get_trained_keras_emulator(emulator, checkpoint_dir, new=False, cube_dir=None, tfrecord_dir=None):
    checkpoint_loc = os.path.join(checkpoint_dir, 'model')
    if new:
        logging.info('Training new emulator')
        if cube_dir is not None:
            assert tfrecord_dir is None
            emulator = train_on_cubes(emulator, *data(cube_dir))
        else:
            assert cube_dir is None
            emulator = train_on_datasets(emulator, *data_from_tfrecords(tfrecord_dir))
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
    for cube_loc in tqdm.tqdm(cube_locs, unit=' cubes saved to tfrecord'):
        logging.debug('Loading and saving cube {}'.format(cube_loc))
        theta, norm_photometry = load_cube(cube_loc)
        ds = tf.data.Dataset.from_tensor_slices((theta, norm_photometry))
        serialized_ds = ds.map(tf_serialize_example)
        tfrecord_loc = cube_loc.rstrip('.hdf5') + '.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(tfrecord_loc)
        writer.write(serialized_ds)
    logging.info('Saved cubes in {} as tfrecords to {}'.format(cube_dir, tfrecord_dir))


    # now write as fixed train/test records
def tfrecords_to_train_test(tfrecord_dir, shards_dir, shards=10):  # will have 1 test shard, otherwise train shards
    tfrecord_locs = glob.glob(os.path.join(tfrecord_dir, 'photometry_simulation_*.tfrecord'))
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_locs) 
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=len(tfrecord_locs),
        block_length=1)
    ds = ds.shuffle(10000)
    for shard_n in tqdm.tqdm(range(shards), unit=' tfrecord shards saved'):
        shard_ds = ds.shard(shards, shard_n)
        # no need to parse, writing straight back out
        if shard_n == 0:
            tfrecord_loc = os.path.join(shards_dir, 'test_{}.tfrecord'.format(shard_n))
        else:
            tfrecord_loc = os.path.join(shards_dir, 'train_{}.tfrecord'.format(shard_n))
        writer = tf.data.experimental.TFRecordWriter(tfrecord_loc)
        writer.write(shard_ds)


def parse_function(example_proto):
    # Create a description of the features.
    feature_description = {
        'theta': tf.io.FixedLenFeature([8], tf.float32),
        'norm_photometry': tf.io.FixedLenFeature([12], tf.float32)
    }
    # Parse the input `tf.Example` proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_example['theta'], parsed_example['norm_photometry']


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
    You can run this to train a new emulator (which will be saved as below)

    To find better model hyperparams, see Google Colab

    Example use:

        python agnfinder/tf_sampling/deep_emulator.py --checkpoint=results/checkpoints/euclid_test
    """
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint', type=str, dest='checkpoint_dir')
    parser.add_argument('--cube', type=str, dest='cube_dir', default='data/cubes/latest')
    # TODO add tfrecords as arg if needed
    args = parser.parse_args()

    tf.config.optimizer.set_jit(True)
    # tf.compat.v1.enable_eager_execution()

    # train_boosted_trees()

    # create_new_emulator
    cube_dir = args.cube_dir
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # convert cubes to tfrecords, one at a time
    # cubes_to_tfrecords(cube_dir, cube_dir)

    # shuffle (out-of-memory) those tfrecords into 9 train and 1 test shards
    # tfrecords_to_train_test(cube_dir, cube_dir, shards=10)
    
    # tfrecord_locs = glob.glob(os.path.join(cube_dir, 'test_0.tfrecord'))
    # print(tfrecord_locs)
    # exit()
    # ds = load_from_tfrecords(tfrecord_locs, batch_size=128)
    # elements = ds.take(/10)
    # for t, p in elements:
        # print(t, p)

    model = tf_model()
    trained_clf = get_trained_keras_emulator(model, checkpoint_dir, new=True, cube_dir=cube_dir)
    # trained_clf = get_trained_keras_emulator(model, checkpoint_dir, new=True, tfrecord_dir=cube_dir)