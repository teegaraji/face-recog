import cv2
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def to_rgb(self, image):
        """Convert BGR (OpenCV default) to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def resize(self, image, size):
        """Resize image to (size, size)."""
        return cv2.resize(image, (size, size))

    def normalize(self, image):
        """Normalize pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0

    def letterbox(self, image, new_shape=640, color=(114, 114, 114)):
        """Resize image with unchanged aspect ratio using padding"""
        shape = image.shape[:2]  # current shape [height, width]
        r = min(new_shape / shape[0], new_shape / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return padded, r, dw, dh

    def preprocess_for_detection(self, image, input_size):
        img = self.to_rgb(image)
        img, r, dw, dh = self.letterbox(img, input_size)
        img = self.normalize(img)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, r, dw, dh

    def preprocess_for_embedding(self, face_img, input_size):
        """Preprocess cropped face for embedding model."""
        img = self.resize(face_img, input_size)
        img = self.normalize(img)
        img = np.expand_dims(img, axis=0)
        return img
