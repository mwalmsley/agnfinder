import datetime

from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint error
from tensorflow.python.saved_model import signature_constants, tag_constants  # pylint error
from tensorflow.python.framework import convert_to_constants
import tensorflow as tf
import numpy as np



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

    # this is a little bit better w/ z1+z4 cubes, 8 params inc redshift
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
        loss='mean_absolute_error',
        metrics=['mean_squared_error'])
    return model


# from agnfinder.tf_sampling import deep_emulator
# avoid for now, to save setting up dockerfile

def checkpoint_to_saved_model(checkpoint_dir, output_dir):
    # model = deep_emulator.tf_model()
    model = tf_model()
    model.load_weights(checkpoint_dir + '/model')  # modifies inplace
    # trained_clf = deep_emulator.get_trained_keras_emulator(model, checkpoint_dir, new=False)
    tf.saved_model.save(model, output_dir)


def model_to_trt(input_model_dir, output_model_dir, conversion_params):
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_model_dir,
        conversion_params=conversion_params)
    converter.convert()
    # not actually optimised yet, just prepped - will optimise at runtime and then cache
    converter.save(output_model_dir)


def create_models(checkpoint_dir, savedmodel_dir, trt_dir, precision_mode):

    checkpoint_to_saved_model(checkpoint_dir, savedmodel_dir)

    # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api-20
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    # print(conversion_params)
    # exit()
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=int(2.1 * 10 ** 9))  # max for laptop, 3GB - overhead
    conversion_params = conversion_params._replace(
        precision_mode=precision_mode)
    # testing on laptop, max_batch does nothing and max_cached is fine 3 or less
    # conversion_params = conversion_params._replace(
    #     max_batch_size=1024)
    # conversion_params = conversion_params._replace(
    #      maximum_cached_engines=1)
    conversion_params = conversion_params._replace(
         minimum_segment_size=3)  # best to be 3 or less

    model_to_trt(savedmodel_dir, trt_dir, conversion_params)


def load_savedmodel(save_dir):
    return tf.saved_model.load(
        save_dir,
        tags=[tag_constants.SERVING])


def load_frozen_savedmodel(save_dir):
    saved_model_loaded = load_savedmodel(save_dir)

    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        graph_func)
    return lambda x: frozen_func(x)[0]  # don't return a list


def benchmark(batches, batch_size, inference_func):
    start_time = datetime.datetime.now()
    for _ in range(batches):
        input_data = tf.constant(np.random.rand(batch_size, 9), dtype=tf.float32)
        _ = inference_func(input_data)
    
    time_elapsed = datetime.datetime.now() - start_time
    time_per_row = time_elapsed.total_seconds() / (batch_size * batches)
    print(f'Done {batches} batches of {batch_size} in {time_elapsed}, {time_per_row}s per row')


def main():

    checkpoint_dir = 'results/checkpoints/latest'
    savedmodel_dir = 'results/checkpoints/latest_savedmodel'
    trt_dir = 'results/checkpoints/latest_trt'
    # savedmodel_dir = '/volume/latest_savedmodel'
    # trt_dir = '/volume/latest_trt'

    precision_mode = "FP16"
    # precision_mode = "FP32"
    print('Creating models')
    create_models(checkpoint_dir, savedmodel_dir, trt_dir, precision_mode)
    # exit()

    savedmodel = load_frozen_savedmodel(savedmodel_dir)
    trt_model = load_frozen_savedmodel(trt_dir)

    batches = 10000
    batch_size = 1024
    # batch_size = 1024

    # print('Using savedmodel')
    # benchmark(batches, batch_size, savedmodel)
    
    # xla_benchmark = tf.function(benchmark, experimental_compile=True)
    # xla_benchmark(batches, batch_size, savedmodel)
    # exit()

    print('Creating TRT')
    # trigger build
    benchmark(1, batch_size, trt_model)

    print('Using TRT')
    # now has build cached
    benchmark(batches, batch_size, trt_model)

    # print('Using savedmodel')
    # benchmark(batches, batch_size, savedmodel)

    # print('Using TRT after TF')
    # run again to check memory allocation
    # benchmark(batches, batch_size, trt_model)


if __name__ == '__main__':

    # print(tf.config.experimental.get_synchronous_execution())
    # set_per_process_memory_fraction(.9))
    # exit()

    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()

    # need to fix my drivers with fresh install including nvidia-cuda-before I can run this

    """
    FP16, laptop

    Done 50000 batches of 1024 in 0:01:01.119907, 1.1937481835937499e-06s per row
    Using TRT
    Done 50000 batches of 1024 in 0:00:31.976154, 6.245342578125e-07s per row

    Creating models
    Using savedmodel
    Done 50000 batches of 128 in 0:00:15.209734, 2.3765209375e-06s per row
    Using TRT
    Done 50000 batches of 128 in 0:00:17.129649, 2.67650765625e-06s per row

    Using savedmodel
    Done 50000 batches of 32 in 0:00:14.705442, 9.19090125e-06s per row
    Using TRT
    Done 50000 batches of 32 in 0:00:16.970270, 1.060641875e-05s per row


    FP16, Zeus

    """