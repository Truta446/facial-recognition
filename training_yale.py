import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(20, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(5, 3000)
lbph = cv2.face.LBPHFaceRecognizer_create(1, 2, 2, 2, 5)

def getImagemComId():
    caminhos = [os.path.join('yalefaces/treinamento', f) for f in os.listdir('yalefaces/treinamento')]
    faces = []
    ids = []

    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')
       imagemNP = np.array(imagemFace, 'uint8')
       id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
       ids.append(id)
       faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print("Training...")
eigenface.train(faces, ids)
eigenface.write('classifierEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('classifierFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classifierLBPHYale.yml')

print("Successful training!")