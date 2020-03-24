import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from agnfinder.tf_sampling import deep_emulator
from agnfinder.prospector import load_photometry


def metric_by_band(metric, y_true, y_pred):
    bands = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h']  # TODO
    return [(band, metric(y_true[:, i], y_pred[:, i])) for i, band in enumerate(bands)]


def maggies_to_mags(maggies):
     # inverse of load_photometry.mags_to_maggies
    return -2.5 * np.log10(maggies)


if __name__ == '__main__':

    """
    Check if the emulator is giving photometry similar to the 'true' forward model.

    Example use:
    python agnfinder/tf_sampling/evaluate_emulator.py --checkpoint-loc results/checkpoints/latest-tf
    """
    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint', type=str, dest='checkpoint_dir', default='results/checkpoints/latest')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    x_train, y_train, x_test, y_test = deep_emulator.data(cube_dir='data/cubes/latest')

    x_train, y_train = x_train[:10000], y_train[:10000]  # for speed
    x_test, y_test = x_test[:10000], y_test[:10000]  # for speed

    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_dir, new=False)

    emulator.evaluate(x_train, y_train, use_multiprocessing=True)
    emulator.evaluate(x_test, y_test, use_multiprocessing=True)

    y_pred = emulator.predict(x_test, use_multiprocessing=True)

    print('Explained Var (max 1): {:.5f}'.format(metrics.explained_variance_score(y_test, y_pred)))

    bands = np.arange(y_train.shape[1])
    med_abs_error = np.array([metrics.median_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])
    rel_med_abs_error = med_abs_error / np.median(y_test, axis=0)
    r2 = np.array([metrics.r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(bands))])

    print('Median absolute error: {}'.format(med_abs_error))
    print('Relative median absolute error: {}'.format([('{:.2f}'.format(100 * x) +r'%') for x in rel_med_abs_error]))
    print('R2 Score: {}'.format(r2))


    # overriding above
    x_test = np.loadtxt('data/cubes/x_test_v2.npy')
    y_test = np.loadtxt('data/cubes/y_test_v2.npy')
    y_pred = emulator.predict(x_test, use_multiprocessing=True)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    all_axes = [ax for row in axes for ax in row]
    for i in bands:
        ax = all_axes[i]
        ax.set_title(i)

        # direct loss target
        # true_log_flux = y_test[:, i]
        # predicted_log_flux = y_pred[:, i]

        # actual flux
        true_flux = deep_emulator.denormalise_photometry(y_test[:, i])
        predicted_flux = deep_emulator.denormalise_photometry(y_pred[:, i])

        # 0-20%, fairly poor? Could be important since it's flux that ultimately matters
        flux_abs_error = np.abs(true_flux - predicted_flux)
        fractional_flux_error = flux_abs_error / true_flux
        # print(flux_abs_error[:10])
        # print(fractional_flux_error[:10])
        # exit()
        

        true_mags = maggies_to_mags(true_flux)
        predicted_mags = maggies_to_mags(predicted_flux)
        # print(true_flux)
        # print(true_mags)
        # print(load_photometry.mags_to_maggies(true_mags))


        # around .1 mags, reasonable but quite a bit higher than speculator? Depends on priors of course
        mags_abs_error = np.abs(true_mags - predicted_mags)
        # 0-1%, fairly good - directly proportional to loss target, so makes sense
        fractional_mag_error = mags_abs_error / true_mags

        # plotting

        # flux errors by flux
        # absolute errors get bigger as the flux gets bigger, which is not surprising
        # ax.scatter(np.log10(true_flux), np.log10(flux_abs_error), alpha=0.3, s=0.1)
        # # ax.set_yscale('log')
        # # ax.set_xscale('log')
        # # ax.set_ylim(0., .2)
        # ax.set_xlabel('Log True Flux')
        # ax.set_ylabel('Log Flux Abs Error')

        # fractional flux errors by flux
        # typical fractional flux errors of around 5%
        # fractional error is fairly constant, which I guess is good
        # ax.scatter(np.log10(true_flux), fractional_flux_error, alpha=0.3, s=0.1)
        # # ax.set_yscale('log')
        # # ax.set_xscale('log')
        # ax.set_ylim(0., .2)
        # ax.set_xlabel('True Flux')
        # ax.set_ylabel('Flux Fractional Error')


        param = x_test[:, 3]
        ax.scatter(param, fractional_flux_error, alpha=0.3, s=0.1)
        ax.set_ylim(0., .2)
        ax.set_xlabel('Param')
        ax.set_ylabel('Flux Fractional Error')

        # mag errors
        # x_max = np.percentile(mags_abs_error, 99.9)
        # x_max = 0.3
        # x = np.linspace(0, x_max)
        # below_x = np.array([np.sum(mags_abs_error < max_error) for max_error in x])
        # ax.plot(x, below_x)

        # mag error by mag
        # ax.scatter(true_mags, mags_abs_error, alpha=0.3, s=0.1)
        # ax.set_ylim(0., .2)
        # ax.set_xlabel('True Mag')
        # ax.set_ylabel('Mag Abs Error')


        # fractional mag error by mag
        # typical fractional mag errors of around 0.5%
        # ax.scatter(true_mags, fractional_mag_error, alpha=0.3, s=0.1)
        # ax.set_ylim(0., .01)
        # ax.set_xlabel('True Mag')
        # ax.set_ylabel('Fractional Mag Error')

        # mag error by redshift
        # redshifts = x_test[:, 0]
        # ax.scatter(redshifts, mags_abs_error, alpha=0.3, s=0.1)
        # ax.set_ylim(0., .2)
        # ax.set_xlabel('Redshift')
        # ax.set_ylabel('Mag Abs Error')


        # param = x_test[:, 2]
        # ax.scatter(param, mags_abs_error, alpha=0.3, s=0.1)
        # ax.set_ylim(0., .2)
        # ax.set_xlabel('Param')
        # ax.set_ylabel('Mag Abs Error')



    fig.tight_layout()
    fig.savefig('temp.png')
