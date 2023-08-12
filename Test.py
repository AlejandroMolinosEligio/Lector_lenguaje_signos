import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from contextlib import redirect_stdout
import os
import tensorflow.keras

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5","Models/labels.txt")

offset = 20 # Tamaño extra para cortar bien la imagen
imgSize = 300 # Tamaño imagen

folder = "Data/C"
counter = 0
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    succes,img = cap.read()
    imgOut = img.copy()
    hands,img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio>1:
            k = imgSize / h
            wCal = math.ceil(k*w)

            if imgCrop.size != 0:

                imgResize = cv2.resize(imgCrop, (wCal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:,wGap:wGap+wCal] = imgResize

                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    with tensorflow.device('/cpu:0'):
                        ## Prediccion
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        
        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            
            if imgCrop.size != 0:

                imgResize = cv2.resize(imgCrop, (imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)  
                imgWhite[hGap:hGap+hCal,:] = imgResize

                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    with tensorflow.device('/cpu:0'):
                        ## Prediccion
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)


        ## Mostrar resultado en pantalla
        
        cv2.rectangle(imgOut, (x-offset,y-offset-50), (x-offset+90,y-offset), (255,0,255),cv2.FILLED)
        cv2.putText(imgOut,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
        cv2.rectangle(imgOut, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),4)

    cv2.imshow("Image",imgOut)
    cv2.waitKey(1)
