import os
import pickle
import tkinter as tk
from tkinter import Button, Label, filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageTk

from app.detection.yolo_detector import YOLOv8FaceDetector
from app.preprocessing.preprocessor import Preprocessor
from app.recognition.embedder import FaceEmbedder
from app.tracking.deepsort_tracker import DeepSortFaceTracker
from app.utils.add_student import add_student

from database.milvus_handler import MilvusHandler  # aktifkan kembali import ini
from database.mysql_handler import MySQLHandler


class FaceAttendanceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.detector = YOLOv8FaceDetector()
        self.embedder = FaceEmbedder()
        self.tracker = DeepSortFaceTracker()
        self.preprocessor = Preprocessor()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_snapshot = Button(
            window, text="Capture", width=10, command=self.capture_face
        )
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        self.btn_add_student = Button(
            window,
            text="Tambah Student",
            width=15,
            command=self.open_add_student_dialog,
        )
        self.btn_add_student.pack(anchor=tk.CENTER, expand=True)

        self.embedding_label = Label(
            window, text="Embedding akan muncul di sini", wraplength=600, justify="left"
        )
        self.embedding_label.pack(pady=10)

        self.absent_student_ids = set()

        self.track_id_to_name = {}

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

    def open_add_student_dialog(self):
        # Input nama, NIM, password
        password = simpledialog.askstring(
            "Password Admin", "Masukkan password admin:", show="*"
        )
        if not password:
            return
        class_name = simpledialog.askstring("Kelas", "Masukkan kode kelas:")
        if not class_name:
            return
        name = simpledialog.askstring("Nama", "Masukkan nama student:")
        if not name:
            return
        nim = simpledialog.askstring("NIM", "Masukkan NIM student:")
        if not nim:
            return
        # Pilih 7 foto
        photo_paths = filedialog.askopenfilenames(
            title="Pilih 7 Foto Wajah",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
        )
        if not photo_paths or len(photo_paths) != 7:
            messagebox.showerror("Error", "Pilih tepat 7 foto wajah!")
            return

        # Simpan foto ke folder (gunakan PIL agar kompatibel)
        from datetime import datetime

        photo_save_dir = "student_photos"
        os.makedirs(photo_save_dir, exist_ok=True)
        saved_paths = []
        for idx, photo_path in enumerate(photo_paths):
            img = Image.open(photo_path)
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation, None)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except Exception as e:
                print("EXIF orientation handling error:", e)
            save_path = os.path.join(
                photo_save_dir,
                f"{nim}_{idx+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
            )
            img.save(save_path)
            saved_paths.append(save_path)

        # Panggil fungsi add_student
        ADMIN_PASSWORD = "admin123"  # Ganti sesuai kebutuhan
        success, msg = add_student(
            name, nim, saved_paths, photo_save_dir, ADMIN_PASSWORD, password, class_name
        )
        messagebox.showinfo("Tambah Student", msg)

    def knn_predict(
        self, embedding, knn_path="student_photos/knn_model.pkl", threshold=0.7
    ):
        if not os.path.exists(knn_path):
            return None, None
        try:
            with open(knn_path, "rb") as f:
                knn_data = pickle.load(f)
        except ModuleNotFoundError as e:
            print(
                "KNN model file is not compatible (possibly contains sklearn object). Please delete and regenerate knn_model.pkl."
            )
            return None, None
        except Exception as e:
            print("Error loading knn_model.pkl:", e)
            return None, None
        embeddings = np.array(knn_data["embeddings"])
        labels = np.array(knn_data["labels"])
        if len(embeddings) == 0:
            return None, None
        # KNN: cosine similarity
        normed = embedding / (np.linalg.norm(embedding) + 1e-8)
        normed_db = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        sims = np.dot(normed_db, normed)
        idx = np.argmax(sims)
        if sims[idx] > threshold:
            return labels[idx], sims[idx]
        return None, None

    def automatic_attendance(self, embedding, threshold=0.7):
        # 1. Search ke Milvus
        milvus_handler = MilvusHandler()
        collection = milvus_handler.collection
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            [embedding.tolist()],
            "embedding",
            search_params,
            limit=1,
            output_fields=["student_id"],
        )
        hits = results[0]
        student_id = None
        if hits and hits[0].distance <= threshold:
            student_id = hits[0].entity.get("student_id")
        else:
            # Coba KNN jika Milvus tidak match
            student_id, sim = self.knn_predict(embedding)
        if not student_id or student_id in self.absent_student_ids:
            return  # Tidak ada match atau sudah absen

        self.absent_student_ids.add(student_id)
        mysql_handler = MySQLHandler()
        # 2. Ambil data student
        conn = mysql_handler.get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT s.name, s.nim, s.photo_path, c.name as class_name, s.class_id FROM students s LEFT JOIN classes c ON s.class_id = c.id WHERE s.id=%s",
            (student_id,),
        )
        student = cursor.fetchone()
        cursor.close()
        conn.close()
        if not student:
            return

        # 3. Insert absensi
        mysql_handler.insert_attendance(student_id, student["class_id"])

        # 4. Tampilkan pop up
        from PIL import Image, ImageTk

        img = Image.open(student["photo_path"])
        img = img.resize((100, 100))
        imgtk = ImageTk.PhotoImage(img)
        top = tk.Toplevel(self.window)
        top.title("Absensi Berhasil")
        tk.Label(
            top,
            text=f"{student['name']} ({student['nim']})\n"
            f"Telah terabsensi di kelas {student['class_name']}",
        ).pack()
        tk.Label(top, image=imgtk).pack()
        top.imgtk = imgtk  # keep reference
        top.after(3000, top.destroy)  # pop up hilang otomatis 3 detik

    def get_name_from_embedding(self, embedding, threshold=0.7):
        milvus_handler = MilvusHandler()
        collection = milvus_handler.collection
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            [embedding.tolist()],
            "embedding",
            search_params,
            limit=1,
            output_fields=["student_id"],
        )
        hits = results[0]
        student_id = None
        if hits and hits[0].distance <= threshold:
            student_id = hits[0].entity.get("student_id")
        else:
            student_id, sim = self.knn_predict(embedding)
        if not student_id:
            return None, None
        mysql_handler = MySQLHandler()
        conn = mysql_handler.get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name FROM students WHERE id=%s", (student_id,))
        student = cursor.fetchone()
        cursor.close()
        conn.close()
        if student:
            return student["name"], student_id
        return None, None

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.current_frame = frame.copy()
            faces, r, dw, dh = self.detector.detect_faces(frame)
            self.current_faces = faces
            self.current_embeddings = []
            h, w = frame.shape[:2]
            embeddings = []

            # Reset mapping setiap frame
            self.track_id_to_name = {}

            for x1, y1, x2, y2, conf in faces:
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
                # Konsisten: align_face dan equalize_histogram
                aligned = self.preprocessor.align_face(face_crop)
                equalized = self.preprocessor.equalize_histogram(aligned)
                embedding = self.embedder.get_embedding(equalized)
                embeddings.append(embedding)

            self.current_embeddings = embeddings

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
            for idx, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                l, t, r_, b = map(int, track.to_ltrb())
                track_id = track.track_id

                # Cek nama dari embedding
                if idx < len(embeddings):
                    name, student_id = self.get_name_from_embedding(embeddings[idx])
                    if name:
                        self.track_id_to_name[track_id] = name
                    else:
                        self.track_id_to_name[track_id] = "Unknown"
                else:
                    self.track_id_to_name[track_id] = "Unknown"

                # Tampilkan ID dan Nama
                label_text = f"ID: {track_id} | {self.track_id_to_name[track_id]}"
                cv2.rectangle(frame, (l, t), (r_, b), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label_text,
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

        if self.current_embeddings:
            self.automatic_attendance(self.current_embeddings[0])
        self.window.after(self.delay, self.update)
        print("Webcam frame shape:", frame.shape)


# Jalankan aplikasi
if __name__ == "__main__":
    mysql_handler = MySQLHandler()
    mysql_handler.create_classes_table()
    mysql_handler.create_students_table()
    mysql_handler.create_attendance_table()

    root = tk.Tk()
    app = FaceAttendanceApp(root, "Face Attendance System")
