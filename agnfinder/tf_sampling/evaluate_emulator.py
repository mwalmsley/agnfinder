import argparse

import numpy as np
from sklearn import metrics

from agnfinder.tf_sampling import deep_emulator


def metric_by_band(metric, y_true, y_pred):
    bands = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h']  # TODO
    return [(band, metric(y_true[:, i], y_pred[:, i])) for i, band in enumerate(bands)]


if __name__ == '__main__':

    """
    Check if the emulator is giving photometry similar to the 'true' forward model.

    Example use:
    /data/miniconda3/envs/agnfinder/bin/python /Data/repos/agnfinder/agnfinder/tf_sampling/evaluate_emulator.py --checkpoint-loc results/checkpoints/latest_tf
    """

    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc')
    args = parser.parse_args()

    checkpoint_loc = args.checkpoint_loc
    _, _, x_test, y_test = deep_emulator.data()

    x_test, y_test = x_test[:10000], y_test[:10000]  # for speed

    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)
    emulator.evaluate(x_test, y_test, use_multiprocessing=True)
    y_pred = emulator.predict(x_test, use_multiprocessing=True)

    print('Explained Var (max 1): {:.5f}'.format(metrics.explained_variance_score(y_test, y_pred)))

    bands = np.arange(12)  # TODO temp
    med_abs_error = np.array([metrics.median_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])
    rel_med_abs_error = med_abs_error / np.median(y_test, axis=0)
    r2 = np.array([metrics.r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])

    print('Median absolute error: {}'.format(med_abs_error))
    print('Relative median absolute error: {}'.format([('{:.2f}'.format(100 * x) +r'%') for x in rel_med_abs_error]))
    print('R2 Score: {}'.format(r2))
