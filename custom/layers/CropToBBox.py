import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.saving.register_keras_serializable(package="layers")
class CropToBBox(layers.Layer):
    def __init__(self, target_shape: tuple[int, int], factor=None, **kwargs):
        super(CropToBBox, self).__init__(**kwargs)
        self.shape = target_shape
        self.f = factor

    def call(self, inputs):
        (threshold, bboxes), images = inputs
        bboxes = self.process_bbox(threshold, bboxes)

        return tf.image.crop_and_resize(images, bboxes, tf.range(tf.shape(bboxes)[0]), self.shape)

    def resize_bbox(self, bbox, factor):
        x1, y1, x2, y2 = tf.unstack(bbox, axis=1)
        new_x1, new_x2 = self.resize_side(x1, x2, factor)
        new_y1, new_y2 = self.resize_side(y1, y2, factor)

        return tf.stack([new_x1, new_y1, new_x2, new_y2], axis=-1)

    def process_bbox(self, threshold, bbox):
        filter = tf.less(threshold, .5)
        filtered = tf.where(filter, tf.constant([0., 1., 0., 1.], dtype=tf.float32), bbox)

        if self.f:
            return self.resize_bbox(filtered, self.f)

        return filtered

    @staticmethod
    def resize_side(small, large, factor):
        side = large - small
        new_side = side * factor
        center = (small + large) / 2
        new_min = tf.clip_by_value(center - new_side / 2, 0., 1.)
        new_max = tf.clip_by_value(center + new_side / 2, 0., 1.)
        return new_min, new_max
