import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import pyttsx3
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Hand detector
detector = HandDetector(maxHands=1)

# Classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "turn on the fan",
    "good morning",
    "I want to leave",
    "Come here",
    "hii everyone",
    "Done",
    "Sleep",
    "Good one"
]

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Speech control
prev_label = ""
last_spoken = time.time()

def text_to_speech(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y-offset)
        y2 = min(img.shape[0], y+h+offset)
        x1 = max(0, x-offset)
        x2 = min(img.shape[1], x+w+offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        confidence = prediction[index]

        print(prediction, index)

        # Apply confidence filter
        if confidence > 0.8:

            label = labels[index]

            cv2.rectangle(imgOutput, (x-offset, y-offset-50),
                          (x-offset+250, y-offset), (255, 0, 255), cv2.FILLED)

            cv2.putText(imgOutput, label, (x, y-26),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x+w+offset, y+h+offset), (255, 0, 255), 4)

            # Speak only if label changes
            if label != prev_label and time.time() - last_spoken > 2:
                text_to_speech(label)
                prev_label = label
                last_spoken = time.time()

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()