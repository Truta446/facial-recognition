import cv2
from os import environ

classifier_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(environ['URL'])  # First webcam available

sample = 1
number_of_samples = 25
identifier = input('Digit your identifier: ')
width, height = 220, 220

print('Capturing faces...')

while True:
    conect, image = camera.read()  # read on webcam

    # Only to machile learning
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = classifier_face.detectMultiScale(
        grey_image, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_image = cv2.resize(
                grey_image[y:y + h, x:x + w], (width, height))
            cv2.imwrite(
                f'images/person.{str(identifier)}.{sample}.jpg', face_image)
            print(f'[Photo {sample} successfully captured!]')
            sample += 1

    cv2.imshow('Face', image)  # Show images from webcam
    cv2.waitKey(1)

    if (sample >= number_of_samples + 1):
        break

camera.release()  # release memory
cv2.destroyAllWindows()
