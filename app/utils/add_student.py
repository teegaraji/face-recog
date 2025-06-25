import os
import pickle

import cv2

from app.detection.yolo_detector import YOLOv8FaceDetector
from app.recognition.embedder import FaceEmbedder
from app.preprocessing.preprocessor import Preprocessor
from database.milvus_handler import MilvusHandler
from database.mysql_handler import MySQLHandler


def add_student(
    name, nim, photo_file, photo_save_dir, admin_password, input_password, class_name
):
    # 1. Validasi password
    if input_password != admin_password:
        return False, "Password salah!"

    # 2. Simpan foto
    os.makedirs(photo_save_dir, exist_ok=True)
    photo_path = os.path.join(photo_save_dir, f"{nim}.jpg")
    photo_file.save(
        photo_path
    )  # Jika pakai Flask/Django, jika Tkinter: gunakan PIL.Image.save

    # 3. Insert ke MySQL
    mysql_handler = MySQLHandler()
    class_id = mysql_handler.get_or_create_class(class_name)
    # Insert student dengan class_id
    conn = mysql_handler.get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO students (name, nim, photo_path, class_id) VALUES (%s, %s, %s, %s)",
        (name, nim, photo_path, class_id),
    )
    conn.commit()
    student_id = cursor.lastrowid
    cursor.close()
    conn.close()

    # 4. Proses embedding
    detector = YOLOv8FaceDetector()
    embedder = FaceEmbedder()
    preprocessor = Preprocessor()
    img = cv2.imread(photo_path)
    if img is None:
        return False, "Foto gagal dibaca, pastikan format gambar benar!"
    print("Image shape:", img.shape)
    faces, r, dw, dh = detector.detect_faces(img)
    print("Detected faces:", faces)
    if not faces:
        return False, "Wajah tidak terdeteksi di foto!"
    x1, y1, x2, y2, _ = faces[0]
    face_crop = img[int(y1) : int(y2), int(x1) : int(x2)]
    # Terapkan align_face dan equalize_histogram
    aligned = preprocessor.align_face(face_crop)
    equalized = preprocessor.equalize_histogram(aligned)
    embedding = embedder.get_embedding(equalized)

    # 5. Insert ke Milvus
    milvus_handler = MilvusHandler()
    collection = milvus_handler.collection
    data = [[student_id], [embedding.tolist()]]
    print(f"Inserted embedding for student_id={student_id} to Milvus")
    collection.insert(data)

    # 6. Simpan ke KNN (pickle)
    knn_path = os.path.join(photo_save_dir, "knn_embeddings.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, "rb") as f:
            knn_data = pickle.load(f)
    else:
        knn_data = {"embeddings": [], "labels": []}
    knn_data["embeddings"].append(embedding.tolist())
    knn_data["labels"].append(student_id)
    with open(knn_path, "wb") as f:
        pickle.dump(knn_data, f)

    return True, "Student berhasil ditambahkan!"
