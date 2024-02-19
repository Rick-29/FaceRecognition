import numpy as np
import cv2

from custom.models.SingleObjectRecognition import Face_Recognizer

model = Face_Recognizer("saved_models/Face_Rec_Model_v3.keras", factor=None) # Adjust factor to change the size of the bounding boxes

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(rgb.dtype)
    preds = model(rgb, verbose=0)
    coords = preds["coords"].numpy()
    threshold = preds["threshold"].numpy()

    if threshold > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(coords[:2]),
                      tuple(coords[2:]),
                      (255, 0, 0), 2)

        name = preds["class_names"][0].decode("utf-8")
        # Controls the text rendered
        cv2.putText(frame, name, tuple(np.add(coords[:2],
                                              [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
