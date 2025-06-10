import cv2
import numpy as np
import onnxruntime as ort

from app.preprocessing.preprocessor import Preprocessor


class YOLOv8FaceDetector:
    def __init__(
        self,
        model_path="saved_models/faceDetection.onnx",
        conf_threshold=0.5,
        input_size=640,
    ):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.preprocessor = Preprocessor()

    def preprocess(self, image):
        return self.preprocessor.preprocess_for_detection(image, self.input_size)

    def detect_faces(self, image):
        h, w = image.shape[:2]
        img_input, r, dw, dh = self.preprocessor.preprocess_for_detection(
            image, self.input_size
        )
        outputs = self.session.run(None, {self.input_name: img_input})[0]
        outputs = np.squeeze(outputs)
        outputs = outputs.T
        faces = []
        for det in outputs:
            conf = det[4]
            if conf >= self.conf_threshold:
                x_center, y_center, width, height = det[0:4]
                scale_factor = 0.85
                width *= scale_factor
                height *= scale_factor

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Simpan koordinat letterbox, unletterbox di GUI
                faces.append((x1, y1, x2, y2, float(conf)))
        return faces, r, dw, dh
