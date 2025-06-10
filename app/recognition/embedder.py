import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

from app.preprocessing.preprocessor import Preprocessor


class FaceEmbedder:
    def __init__(
        self, model_path="saved_models/embedding_model_savedmodel", input_size=160
    ):
        # Gunakan TFSMLayer untuk SavedModel
        self.model = TFSMLayer(model_path, call_endpoint="serving_default")
        self.input_size = input_size
        self.preprocessor = Preprocessor()

    def preprocess(self, face_img):
        return self.preprocessor.preprocess_for_embedding(face_img, self.input_size)

    def get_embedding(self, face_img):
        preprocessed = self.preprocess(face_img)
        embedding_dict = self.model(preprocessed)
        embedding_tensor = list(embedding_dict.values())[0]
        return embedding_tensor.numpy()[0]
