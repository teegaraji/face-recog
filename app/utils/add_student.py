import os
import pickle

import cv2
import numpy as np
from PIL import ExifTags, Image

from app.detection.yolo_detector import YOLOv8FaceDetector
from app.preprocessing.preprocessor import Preprocessor
from app.recognition.embedder import FaceEmbedder
from database.milvus_handler import MilvusHandler
from database.mysql_handler import MySQLHandler


def augment_image(img):
    """Augmentasi sederhana: flip, brightness, dsb."""
    aug_imgs = [img]
    # Horizontal flip
    aug_imgs.append(cv2.flip(img, 1))
    # Brightness up/down
    for alpha in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        aug_imgs.append(bright)
        aug_imgs.append(cv2.flip(bright, 1))
    # Gaussian blur
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    aug_imgs.append(blur)
    aug_imgs.append(cv2.flip(blur, 1))
    # Contrast
    for alpha in [0.7, 1.3]:
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        aug_imgs.append(contrast)
        aug_imgs.append(cv2.flip(contrast, 1))
    # Total 8 augmentasi (1 ori + 7 augmentasi) per gambar
    return aug_imgs[:2]  # Ambil 2 per gambar agar total 14 (7x2)


def fix_exif_orientation(img_path):
    img = Image.open(img_path)
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
    return np.array(img.convert("RGB"))


def add_student(
    name, nim, photo_paths, photo_save_dir, admin_password, input_password, class_name
):
    # 1. Validasi password
    if input_password != admin_password:
        return False, "Password salah!"

    # 2. Simpan foto (sudah dilakukan di GUI, photo_paths sudah berisi 7 path)
    if len(photo_paths) != 7:
        return False, "Jumlah foto harus 7!"

    # 3. Insert ke MySQL
    mysql_handler = MySQLHandler()
    class_id = mysql_handler.get_or_create_class(class_name)
    conn = mysql_handler.get_mysql_connection()
    cursor = conn.cursor()
    # Simpan path foto pertama sebagai photo_path utama
    cursor.execute(
        "INSERT INTO students (name, nim, photo_path, class_id) VALUES (%s, %s, %s, %s)",
        (name, nim, photo_paths[0], class_id),
    )
    conn.commit()
    student_id = cursor.lastrowid
    cursor.close()
    conn.close()

    # 4. Proses embedding untuk semua augmentasi dari 7 foto
    detector = YOLOv8FaceDetector()
    embedder = FaceEmbedder()
    preprocessor = Preprocessor()
    all_embeddings = []
    for photo_path in photo_paths:
        # Perbaiki orientasi EXIF sebelum deteksi wajah
        img = fix_exif_orientation(photo_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            return False, f"Foto gagal dibaca: {photo_path}"
        faces, r, dw, dh = detector.detect_faces(img)
        if not faces:
            return False, f"Wajah tidak terdeteksi di foto: {photo_path}"
        x1, y1, x2, y2, _ = faces[0]
        face_crop = img[int(y1) : int(y2), int(x1) : int(x2)]
        # Konsisten: align_face dan equalize_histogram
        aligned = preprocessor.align_face(face_crop)
        equalized = preprocessor.equalize_histogram(aligned)
        # Augmentasi
        aug_imgs = augment_image(equalized)
        for aug_img in aug_imgs:
            embedding = embedder.get_embedding(aug_img)
            all_embeddings.append(embedding)

    # 5. Insert semua embedding ke Milvus
    milvus_handler = MilvusHandler()
    collection = milvus_handler.collection
    data = [
        [student_id] * len(all_embeddings),
        [emb.tolist() for emb in all_embeddings],
    ]
    print(
        f"Inserted {len(all_embeddings)} embeddings for student_id={student_id} to Milvus"
    )
    collection.insert(data)

    # 6. Simpan ke KNN (pickle)
    knn_path = os.path.join(photo_save_dir, "knn_model.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, "rb") as f:
            knn_data = pickle.load(f)
    else:
        knn_data = {"embeddings": [], "labels": []}
    for emb in all_embeddings:
        knn_data["embeddings"].append(emb.tolist())
        knn_data["labels"].append(student_id)
    with open(knn_path, "wb") as f:
        pickle.dump(knn_data, f)

    return True, f"Student berhasil ditambahkan! ({len(all_embeddings)} embedding)"
    return True, f"Student berhasil ditambahkan! ({len(all_embeddings)} embedding)"
