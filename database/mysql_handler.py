import mysql.connector


class MySQLHandler:
    def __init__(
        self,
        host="localhost",
        user="root",
        password="root",
        database="face_presensee_db",
    ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def get_mysql_connection(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def create_classes_table(self):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS classes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) UNIQUE
            )
            """
        )
        conn.commit()
        cursor.close()
        conn.close()

    def create_students_table(self):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                nim VARCHAR(20) UNIQUE,
                photo_path VARCHAR(255),
                class_id INT,
                FOREIGN KEY (class_id) REFERENCES classes(id)
            )
            """
        )
        conn.commit()
        cursor.close()
        conn.close()

    def alter_students_add_class_id(self):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        # Tambahkan kolom class_id jika belum ada
        cursor.execute(
            """
            ALTER TABLE students 
            ADD COLUMN IF NOT EXISTS class_id INT,
            ADD FOREIGN KEY (class_id) REFERENCES classes(id)
            """
        )
        conn.commit()
        cursor.close()
        conn.close()

    def create_attendance_table(self):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT,
                class_id INT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (class_id) REFERENCES classes(id)
            )
            """
        )
        conn.commit()
        cursor.close()
        conn.close()

    def insert_attendance(self, student_id, class_id):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO attendance (student_id, class_id) VALUES (%s, %s)",
            (student_id, class_id),
        )
        conn.commit()
        cursor.close()
        conn.close()

    def get_or_create_class(self, class_name):
        conn = self.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM classes WHERE name=%s", (class_name,))
        result = cursor.fetchone()
        if result:
            class_id = result[0]
        else:
            cursor.execute("INSERT INTO classes (name) VALUES (%s)", (class_name,))
            conn.commit()
            class_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return class_id
