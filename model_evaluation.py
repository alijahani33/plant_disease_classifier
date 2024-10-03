import tensorflow as tf
from typing import Tuple
from fastapi import HTTPException
from .config import MODEL_SAVE_PATH

def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dateset) -> Tuple[float, float]:

    try:
        scores = model.evaluate(test_dataset)
        print(f"Test score: {scores[0]}, Test accuracy:{scores[1]}")
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving Model"(e))

def save_model(MODEL_SAVE_PATH) ->None:
    try:
        tf.keras.Model.save(MODEL_SAVE_PATH)
        print(f"Model saved at {MODEL_SAVE_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error saving model:{e}")
