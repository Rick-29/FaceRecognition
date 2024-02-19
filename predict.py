from pprint import pprint

import tensorflow as tf
import argparse
import requests

from custom.models.SingleObjectRecognition import Face_Recognizer


def load_img(path: str):
    return tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(path)
    )

def load_img_from_url(url: str):
    return tf.io.decode_image(
       requests.get(url).content, channels=3 
    )

def load_image(path: str, url: bool = False):
    return load_img_from_url(path) if url else load_img(path)


def main():
    parser = argparse.ArgumentParser(prog="Face Tracker", description="Calls a model to predict the location of the face in a given image and try to predict the person's name from a pretrained list")
    parser.add_argument("--model_path", "-m", type=str, required=False, default="saved_models/Face_Rec_Model_v3.keras", help="Optional: Path to the model to use")
    parser.add_argument("--path", "-p", type=str, required=True, help= "Path to the image to make predictions on")
    parser.add_argument("--url", "-u", type=bool, required=False, default=False, help="Set this to 'True' if the path given is an url")
    parser.add_argument("--factor", "-f", type=float, required=False, default=1, help="Selects the factor appliyed to the bounding box to increase or decrease it's relative size")

    args = parser.parse_args()
    
    model = Face_Recognizer(args.model_path, factor=args.factor)
    preds = model(load_image(args.path, args.url))
    print("\n\n", "-"* 50)
    print("Predictions")
    print()
    pprint(preds)
    
if __name__ == "__main__":
    main()