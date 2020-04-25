import cv2
from os import environ

detect_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read('classifierEigen.yml')
# recognizer.read('classifierEigenYale.yml')
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(environ['URL'])  # First webcam available

while True:
    conect, image = camera.read()
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_face.detectMultiScale(
        grey_image, scaleFactor=1.5, minSize=(30, 30))

    for (x, y, w, h) in detected_faces:
        face_image = cv2.resize(grey_image[y:y + h, x:x + w], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        id, reliability = recognizer.predict(face_image)

        name = ''

        if id == 1:
            name = 'Juan'
        elif id == 2:
            name = 'Tamiris'
        else:
            name = 'Desconhecido'

        cv2.putText(image, name, (x, y + (h + 30)), font, 2, (0, 0, 255))
        cv2.putText(image, str(reliability),
                    (x, y + (h + 50)), font, 1, (0, 0, 255))

    cv2.imshow('Face', image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
