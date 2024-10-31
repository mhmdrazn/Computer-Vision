import cv2
import numpy as np
import os
from datetime import datetime

class FaceRecognition:
    def __init__(self):
        # Inisialisasi detector wajah dan landmark
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Direktori untuk menyimpan dataset wajah
        self.dataset_path = "face_dataset"
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            
        # Dictionary untuk menyimpan label dan nama
        self.labels = {}
        self.current_label = 0
        
    def preprocess_face(self, image):
        """
        Memproses gambar untuk deteksi wajah yang lebih baik
        """
        # Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Equalizer histogram untuk pencahayaan yang lebih baik
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def detect_face(self, image):
        """
        Mendeteksi wajah dalam gambar
        """
        gray = self.preprocess_face(image)
        
        # Deteksi wajah dengan parameter yang dioptimalkan
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    def capture_face_data(self, name, num_samples=200):
        """
        Mengambil sampel wajah untuk training
        """
        cap = cv2.VideoCapture(0)
        count = 0
        
        # Buat folder untuk individu
        person_path = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        
        print(f"Mengambil sampel wajah untuk {name}...")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces, gray = self.detect_face(frame)
            
            for (x, y, w, h) in faces:
                # Gambar kotak di sekitar wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Simpan wajah yang terdeteksi
                face_roi = gray[y:y+h, x:x+w]
                face_filename = os.path.join(person_path, f"{name}_{count}.jpg")
                cv2.imwrite(face_filename, face_roi)
                count += 1
                
                # Tampilkan progress
                cv2.putText(frame, f"Captured: {count}/{num_samples}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Capture Face Data", frame)
            
            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Tambahkan ke dictionary labels
        self.labels[self.current_label] = name
        self.current_label += 1
        
        print(f"Selesai mengambil sampel untuk {name}")
        
    def train_recognizer(self):
        """
        Melatih face recognizer dengan dataset yang ada
        """
        faces = []
        labels = []
        
        # Load semua gambar training
        for label, name in self.labels.items():
            person_path = os.path.join(self.dataset_path, name)
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize untuk konsistensi
                face_img = cv2.resize(face_img, (100, 100))
                
                faces.append(face_img)
                labels.append(label)
        
        print("Melatih face recognizer...")
        self.face_recognizer.train(faces, np.array(labels))
        print("Training selesai!")
        
        # Simpan model
        self.face_recognizer.save("face_recognizer_model.yml")
        
    def recognize_faces(self):
        """
        Melakukan recognition wajah secara real-time
        """
        cap = cv2.VideoCapture(0)
        
        print("Memulai face recognition. Tekan 'q' untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces, gray = self.detect_face(frame)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Prediksi wajah
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Gambar hasil
                if confidence < 70:  # Threshold confidence
                    name = self.labels.get(label, "Unknown")
                    confidence = round(100 - confidence)
                    color = (0, 255, 0)  # Hijau untuk match yang baik
                else:
                    name = "Unknown"
                    confidence = 0
                    color = (0, 0, 255)  # Merah untuk unknown
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({confidence}%)", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Contoh penggunaan
def main():
    face_rec = FaceRecognition()
    
    while True:
        print("\nMenu Face Recognition:")
        print("1. Tambah data wajah baru")
        print("2. Train ulang model")
        print("3. Mulai recognition")
        print("4. Keluar")
        
        choice = input("Pilih menu (1-4): ")
        
        if choice == '1':
            name = input("Masukkan nama: ")
            face_rec.capture_face_data(name)
        elif choice == '2':
            face_rec.train_recognizer()
        elif choice == '3':
            face_rec.recognize_faces()
        elif choice == '4':
            break
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()