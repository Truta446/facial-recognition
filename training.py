import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(20, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(5, 3000)
lbph = cv2.face.LBPHFaceRecognizer_create(1, 2, 2, 2, 5)


def getImageWithId():
    paths = [os.path.join('images', f) for f in os.listdir('images')]

    faces = []
    ids = []

    for path in paths:
        face_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        identifier = int(os.path.split(path)[-1].split('.')[1])
        ids.append(identifier)
        faces.append(face_image)

    return np.array(ids), faces


array_id, array_face = getImageWithId()

print('Training...')

eigenface.train(array_face, array_id)
eigenface.write('classifierEigen.yml')

fisherface.train(array_face, array_id)
fisherface.write('classifierFisher.yml')

lbph.train(array_face, array_id)
lbph.write('classifierLBPH.yml')

print('Successful training!')


