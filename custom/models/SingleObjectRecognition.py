import tensorflow as tf

from custom.layers.CropToBBox import CropToBBox
from custom.layers.PostProcessor import PostProcessor


class Face_Recognizer(tf.keras.Model):
    def __init__(self, model, factor: float | None = None, **kwargs):
        super(Face_Recognizer, self).__init__(**kwargs)

        self.model = model
        crop2bbox = CropToBBox(target_shape=(224, 224), factor=factor)
        if type(model) == str:
            self.model = tf.keras.models.load_model(model,
                                                    custom_objects={"CropToBBox": crop2bbox,
                                                                    "PostProcessor": PostProcessor})

    def call(self, x,  return_image: bool = False, **kwargs):
        if "training" in kwargs.keys():
            kwargs.pop("training")

        if not isinstance(x, tf.data.Dataset):
            if type(x) == str:
                x = tf.expand_dims(self.preprocess_image(x), 0)
            if tf.rank(x) == 3:
                x = tf.expand_dims(x, 0)
        else:
            if isinstance(x, str):
                x = self.preprocess_image(x)

            elif tf.rank(x) == 3:
                x = tf.expand_dims(x, axis=0)

        shape = tf.cast(tf.squeeze(tf.shape(x)[1:]), tf.float32)

        predictions = self.model.predict(x, **kwargs)
        classification = predictions[0]
        threshold = tf.squeeze(predictions[1][0])
        normalized_coords = predictions[1][1]

        x1, y1, x2, y2 = tf.unstack(normalized_coords, axis=1)
        x1, x2 = tf.math.multiply(x1, shape[1]), tf.math.multiply(x2, shape[1])
        y1, y2 = tf.math.multiply(y1, shape[0]), tf.math.multiply(y2, shape[0])
        coords = tf.cast(tf.stack([x1, y1, x2, y2], axis=1), tf.int32)

        outputs = {
            **classification,
            "threshold": threshold,
            "normalized_coords": tf.squeeze(normalized_coords),
            "coords": tf.squeeze(coords)
        }
        if return_image:
            return outputs, x
        return outputs

    def single_pred(self, x, return_face: bool = False, **kwargs):
        predictions = self.call(x, return_face, **kwargs)
        if not return_face:
            return predictions
        preds = predictions[0]
        coords = preds["coords"]

        image = tf.squeeze(predictions[1]).numpy()
        return {
            **preds,
            "face": image[coords[1]:coords[3], coords[0]:coords[2], :]
        }

    @staticmethod
    def preprocess_image(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        return image