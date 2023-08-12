import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20 # Tamaño extra para cortar bien la imagen
imgSize = 300 # Tamaño imagen

folder = "Data/Z"
counter = 0

while True:
    succes,img = cap.read()
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
        
        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            if imgCrop.size != 0:

                imgResize = cv2.resize(imgCrop, (imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)  
                imgWhite[hGap:hGap+hCal,:] = imgResize




        #cv2.imshow("ImageCrop",imgCrop)
        #cv2.imshow("ImageCrop",imgWhite)



    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key==ord("s"):

        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print("Imagen guardada",counter)