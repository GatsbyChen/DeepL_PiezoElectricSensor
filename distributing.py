import tensorflow as tf 


def build_strategy(n_gpus):
    gpus = tf.config.list_physical_devices('GPU')
    print('Available GPUs:', gpus)
    tf.config.set_visible_devices(gpus[:n_gpus], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print('Visible logical gpus:', logical_gpus)

    strategy = tf.distribute.MirroredStrategy(logical_gpus)
    return strategy

gpus = tf.config.experimental.list_physical_devices('CPU')
print(gpus)
