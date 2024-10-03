import tensorflow as tf
from typing import Any, Dict
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import HTTPException
from .config import MODEL_SAVE_PATH, N_CLASSES, IMAGE_SIZE


def load_model() -> tf.keras.Model:
    try:
        return tf.keras.load_model(MODEL_SAVE_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

def make_prediction(model: tf.keras.Model, dataset: tf.keras.Dataset) -> Any:

    prediction = model.predict(dataset)
    return prediction


def read_file_as_file(data: bytes) -> np.ndarray:

    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


def predict(model: tf.keras.Model, image_path: str) -> dict[str, float | Any]:
    try:

        img = keras_image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = N_CLASSES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        return {
            'CLASS': pred_class,
            'Confidence': float(confidence)
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)