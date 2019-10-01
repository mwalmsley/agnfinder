import tensorflow as tf
import numpy as np
from sklearn import metrics

from agnfinder.tf_sampling import deep_emulator


def metric_by_band(metric, y_true, y_pred):
    bands = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h']  # TODO
    return [(band, metric(y_true[:, i], y_pred[:, i])) for i, band in enumerate(bands)]


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = deep_emulator.data()

    x_test, y_test = x_test[:10000], y_test[:10000]  # for speed

    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    emulator.evaluate(x_test, y_test, use_multiprocessing=True)

    y_pred = emulator.predict(x_test, use_multiprocessing=True)
    
    print('Explained Var (max 1): {:.5f}'.format(metrics.explained_variance_score(y_test, y_pred)))

    bands = np.arange(12)  # TODO temp
    med_abs_error = np.array([metrics.median_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])
    rel_med_abs_error = med_abs_error / np.median(y_test, axis=0)
    r2 = np.array([metrics.r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])

    print(med_abs_error)
    print([('{:.2f}'.format(100 * x) +r'%') for x in rel_med_abs_error])
    print(r2)
