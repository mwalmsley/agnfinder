import datetime

from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint error
from tensorflow.python.saved_model import signature_constants, tag_constants  # pylint error
from tensorflow.python.framework import convert_to_constants
import tensorflow as tf
import numpy as np

from agnfinder.tf_sampling import deep_emulator

def checkpoint_to_saved_model(checkpoint_dir, output_dir):
    model = deep_emulator.tf_model()
    trained_clf = deep_emulator.get_trained_keras_emulator(model, checkpoint_dir, new=False)
    tf.saved_model.save(trained_clf, output_dir)


def model_to_trt(input_model_dir, output_model_dir, conversion_params=None):
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_model_dir,
        conversion_params=conversion_params)
    converter.convert()
    # not actually optimised yet, just prepped - will optimise at runtime and then cache
    converter.save(output_model_dir)


def create_models(checkpoint_dir, savedmodel_dir, trt_dir, precision_mode):

    checkpoint_to_saved_model(checkpoint_dir, savedmodel_dir)

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    conversion_params = conversion_params._replace(
         maximum_cached_engines=100)

    model_to_trt(savedmodel_dir, trt_dir, conversion_params=conversion_params)


def load_savedmodel(save_dir):
    return tf.saved_model.load(
        savedmodel_dir,
        tags=[tag_constants.SERVING])


def load_trt_model(savedmodel_dir, trt_dir):
    saved_model_loaded = load_savedmodel(savedmodel_dir)

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



if __name__ == '__main__':

    checkpoint_dir = 'results/checkpoints/latest'
    savedmodel_dir = 'results/checkpoints/latest_savedmodel'
    trt_dir = 'results/checkpoints/latest_trt'

    # precision_mode = "FP16"
    precision_mode = "FP32"
    create_models(checkpoint_dir, savedmodel_dir, trt_dir, precision_mode)

    savedmodel = load_savedmodel(savedmodel_dir)
    trt_model = load_trt_model(savedmodel_dir, trt_dir)

    batches = 10000
    batch_size = 512  # on laptop: 6us at 32, 1.6us at 512, 1.3us at 512 and FP16

    benchmark(batches, batch_size, savedmodel)
    
    # xla_benchmark = tf.function(benchmark, experimental_compile=True)
    # xla_benchmark(batches, batch_size, savedmodel)

    benchmark(batches, batch_size, trt_model)
