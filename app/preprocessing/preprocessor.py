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

    def align_face(self, image):
        """Passthrough: tidak ada alignment berbasis landmark."""
        return image

    def equalize_histogram(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

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
        shape = self.predictor(img_rgb, dets[0])
        # Ambil koordinat mata kiri dan kanan dari landmark
        left_eye = (shape.part(36).x, shape.part(36).y)
        right_eye = (shape.part(45).x, shape.part(45).y)

        # Hitung sudut rotasi
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Tentukan pusat antara kedua mata
        eyes_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        # Buat matriks rotasi
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        aligned = cv2.warpAffine(
            img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.INTER_CUBIC
        )
        if aligned.dtype != np.uint8:
            aligned = aligned.astype(np.uint8)
        if len(aligned.shape) == 2:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
        print("align_face output dtype:", aligned.dtype, "shape:", aligned.shape)
        return aligned

    def equalize_histogram(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

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
