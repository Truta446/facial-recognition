import cv2
import os
import numpy as np
from PIL import Image

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer.read("classifierEigenYale.yml")
# recognizer = cv2.face.FisherFaceRecognizer_create()
# recognizer.read("classifierFisherYale.yml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifierLBPHYale.yml")

total_hits = 0
hit_percentage = 0.0
total_reliability = 0.0

paths = [os.path.join('yalefaces/teste', f) for f in os.listdir('yalefaces/teste')]

for path_image in paths:
  image_face = Image.open(path_image).convert('L')
  image_face_np = np.array(image_face, 'uint8')
  detect_faces = face_detector.detectMultiScale(image_face_np)

  for (x, y, w, h) in detect_faces:
    provided_id, reliability = recognizer.predict(image_face_np)
    current_id = int(os.path.split(path_image)[1].split(".")[0].replace("subject", ""))
    print(f'{str(current_id)} was classified as {str(provided_id)} with a reliability of {str(reliability)}')

    if provided_id == current_id:
      total_hits += 1
      total_reliability += reliability

    # cv2.rectangle(image_face_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("Face", image_face_np)
    # cv2.waitKey(1000)

hit_percentage = ((total_hits / 30) * 100)
total_reliability /= total_hits
print(f'Total Hits: {str(total_hits)}/30')
print(f'Hit Percentage: {hit_percentage}')
print(f'Total Reliability: {str(total_reliability)}')
