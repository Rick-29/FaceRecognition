import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.saving.register_keras_serializable(package="layers")
class PostProcessor(layers.Layer):
    def __init__(self, class_names, **kwargs):
        super(PostProcessor, self).__init__(**kwargs)
        self.class_names = class_names

    def call(self, x):
        class_ids = tf.argmax(x, axis=-1)
        class_names = tf.gather(self.class_names, class_ids)
        return {
            "predictions": x,
            "class_ids": class_ids,
            "class_names": class_names
        }
