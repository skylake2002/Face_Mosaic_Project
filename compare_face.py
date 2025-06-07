import face_recognition
import numpy as np
import sys
import os

known_face_encoding = np.load("my_face.npy")  # 사진 경로

def compare_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) == 0:
        print("얼굴 인식 안됨.")
        return

    for location, encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_face_encoding], encoding, tolerance=0.4)
        if match[0]:
            top, right, bottom, left = location
            print(f"USER_FACE: {left},{top},{right},{bottom}")  # ← 이 형식으로 출력

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("명령어 터미널에 입력: python compare_face.py compare_face.jpg")
        sys.exit(1)

    compare_faces(sys.argv[1])
