import tensorflow as tf

from agnfinder.tf_sampling import deep_emulator

if __name__ == '__main__':

    # tf.enable_eager_execution()

    # checkpoint_loc = 'results/checkpoints/trained_deep_emulator_oct/latest'
    # checkpoint_loc = 'results/trained_deep_emulator_oct'
    # checkpoint_loc = 'results/trained_deep_emulator_oct/checkpoint'
    # checkpoint_loc = 'results/checkpoints/trained_deep_emulator'
    # blank_emulator = deep_emulator.tf_model()
    # emulator = deep_emulator.get_trained_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    fresh_emulator = deep_emulator.tf_model()
    # checkpoint_loc = 'results/checkpoints/trained_deep_emulator'  # must match saved checkpoint of emulator
    # print('Fresh model config:')
    # print(fresh_emulator.get_config())

    x_train, y_train, x_test, y_test = deep_emulator.data()
    # fresh_emulator.evaluate(x_train, y_train, use_multiprocessing=True)  # should be bad

    # status = fresh_emulator.load_weights('results/checkpoints/weights_only/latest_tf')
    # status.assert_consumed()
    # emulator = fresh_emulator
    # emulator = deep_emulator.get_trained_emulator(fresh_emulator, checkpoint_loc, new=False)

    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'
    emulator = deep_emulator.get_trained_keras_emulator(fresh_emulator, checkpoint_loc, new=False)

    emulator.evaluate(x_train, y_train, use_multiprocessing=True)
    emulator.evaluate(x_test, y_test, use_multiprocessing=True)
