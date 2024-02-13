from PIL import Image

import tensorflow as tf
import numpy as np
import cv2

from Custom_Objects.models.image.SingleObjectRecognition import Face_Recognizer

model = Face_Recognizer("saved_models/Face_Rec_Model_v3.keras")


def downscale_image_tf(image):
    shape = image.shape[:-1]
    img_d = tf.image.resize(image, size=(shape[0] // 6, shape[1] // 6))
    return tf.clip_by_value(tf.image.resize(img_d, size=shape), 0, 255).numpy().astype(np.uint8)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    preds = model.single_pred(rgb, return_face=True, verbose=0)
    coords = preds["coords"].numpy()
    threshold = preds["threshold"].numpy()

    frame = downscale_image_tf(frame)
    if threshold > 0.5:
        frame[coords[1]:coords[3], coords[0]:coords[2], :] = cv2.cvtColor(preds["face"], cv2.COLOR_RGB2BGR)

        name = preds["class_names"][0].decode("utf-8").replace("_", " ")
        # Controls the text rendered
        cv2.putText(frame, name, tuple(np.add(coords[:2],
                                              [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
