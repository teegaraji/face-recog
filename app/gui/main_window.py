import tkinter as tk
from tkinter import Button, Label

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.detection.yolo_detector import YOLOv8FaceDetector
from app.recognition.embedder import FaceEmbedder
from app.tracking.deepsort_tracker import DeepSortFaceTracker


class FaceAttendanceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.detector = YOLOv8FaceDetector()
        self.embedder = FaceEmbedder()
        self.tracker = DeepSortFaceTracker()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_snapshot = Button(
            window, text="Capture", width=10, command=self.capture_face
        )
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.embedding_label = Label(
            window, text="Embedding akan muncul di sini", wraplength=600, justify="left"
        )
        self.embedding_label.pack(pady=10)

        self.current_frame = None
        self.current_faces = []
        self.current_embeddings = []

        self.delay = 15
        self.update()
        self.window.mainloop()

    def capture_face(self):
        """Ambil embedding wajah pertama yang terdeteksi"""
        if self.current_faces and self.current_embeddings:
            embedding = self.current_embeddings[0]  # ambil wajah pertama
            embedding_str = ", ".join([f"{val:.4f}" for val in embedding[:10]]) + "..."
            self.embedding_label.config(text=f"Embedding: [{embedding_str}]")
        else:
            self.embedding_label.config(text="Tidak ada wajah yang terdeteksi.")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.current_frame = frame.copy()
            faces, r, dw, dh = self.detector.detect_faces(frame)
            self.current_faces = faces
            self.current_embeddings = []
            h, w = frame.shape[:2]
            embeddings = []

            for x1, y1, x2, y2, conf in faces:
                # Unletterbox koordinat ke frame asli
                x1u = int((x1 - dw) / r)
                y1u = int((y1 - dh) / r)
                x2u = int((x2 - dw) / r)
                y2u = int((y2 - dh) / r)
                x1u = max(0, min(x1u, w - 1))
                y1u = max(0, min(y1u, h - 1))
                x2u = max(0, min(x2u, w - 1))
                y2u = max(0, min(y2u, h - 1))

                face_crop = frame[y1u:y2u, x1u:x2u]
                if face_crop.size == 0:
                    embeddings.append(np.zeros((128,)))
                    continue
                embedding = self.embedder.get_embedding(face_crop)
                embeddings.append(embedding)

            self.current_embeddings = embeddings

            # Gunakan bounding box unletterbox untuk tracker
            detections = [
                [
                    [
                        int((x1 - dw) / r),
                        int((y1 - dh) / r),
                        int((x2 - dw) / r),
                        int((y2 - dh) / r),
                    ],
                    conf,
                ]
                for (x1, y1, x2, y2, conf) in faces
            ]
            tracks = self.tracker.update(detections, embeddings, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                l, t, r_, b = map(int, track.to_ltrb())
                track_id = track.track_id
                cv2.rectangle(frame, (l, t), (r_, b), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Convert to ImageTk
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # keep reference

        self.window.after(self.delay, self.update)
        print("Webcam frame shape:", frame.shape)


# Jalankan aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root, "Face Attendance System")
