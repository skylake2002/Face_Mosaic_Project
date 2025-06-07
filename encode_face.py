import face_recognition
import numpy as np
import sys

def encode_face(image_path: str, save_path: str = "사 용 자 얼 굴 사 진 경 로 (ex : my_face.jpg)"): # 사진 경로 
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("얼굴 인식 안됨.")
        return

    np.save(save_path, encodings[0])
    print(f"저정 :  {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("명령어 터미널에 입력: python encode_face.py my_face.jpg")
        sys.exit(1)

    encode_face(sys.argv[1])
