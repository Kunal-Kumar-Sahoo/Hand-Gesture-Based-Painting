import cv2
import mediapipe as mp
import os
import numpy as np
import HandTrackingModule as htm

################################################
#Configurable variables
brushThickness = 15
eraserThickness = 100
################################################


folderPath = "Header Files"
myList = os.listdir(folderPath)
# print(myList)  Output: ['4.png', '1.png', '3.png', '2.png']

overlayList = []  
for imgPath in myList:
	image = cv2.imread(f"{folderPath}/{imgPath}")
	overlayList.append(image)
# print(overlayList)

header = overlayList[1]
drawColour = (255, 0, 255)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionConf=0.85)
xp, yp = 0, 0
imageCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
	# Import Image
	_, image = cap.read()
	image = cv2.flip(image, 1)

	# Find Hand Landmarks
	image = detector.findHands(image)
	landmarkList = detector.findPosition(image, draw=False)

	if len(landmarkList) != 0:
		# print(landmarkList)

		# Tip of  index and middle finger
		x1, y1 = landmarkList[8][1:]
		x2, y2 = landmarkList[12][1:]


		# Check which fingers are up ?
		fingers = detector.fingersUp()
		# print(fingers)

		# Selection mode : 2 fingers are up
		if fingers[1] and fingers[2]:
			# print("Selection mode")
			xp, yp = 0, 0
			if y1 < 125:
				if 250 < x1 < 450:
					header = overlayList[1] 
					drawColour = (255, 0, 255) # Purple
				elif 550 < x1 < 750:
					header = overlayList[3]
					drawColour = (255, 0, 0) # Blue
				elif 800 < x1 < 950:
					header = overlayList[2]
					drawColour = (0, 255, 0) # Green
				elif 1050 < x1 < 1200:
					header = overlayList[0]
					drawColour = (0, 0, 0) # Black

			cv2.rectangle(image, (x1, y1-15), (x2, y2+15), drawColour, cv2.FILLED)
		# Drawing mode : Index finger is up
		if fingers[1] and not fingers[2]:
			cv2.circle(image, (x1, y1), 15, drawColour, cv2.FILLED)
			# print("Drawing mode")
			if xp == 0 and yp == 0:
				xp, yp = x1, y1

			if  drawColour == (0, 0, 0):
				cv2.line(image, (xp, yp), (x1, y1), drawColour, eraserThickness)
				cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColour, eraserThickness)

			cv2.line(image, (xp, yp), (x1, y1), drawColour, brushThickness)
			cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColour, brushThickness)
			xp, yp = x1, y1

	imageGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
	_, imageInverse = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY_INV)
	imageInverse = cv2.cvtColor(imageInverse, cv2.COLOR_GRAY2BGR)
	image = cv2.bitwise_and(image, imageInverse)
	image = cv2.bitwise_or(image, imageCanvas)

	image[0:125, 0:1280] = header # Parcing the image as it is a matrix
	# image = cv2.addWeighted(image, 0.5, imageCanvas, 0.5, 0)
	cv2.imshow("Frontend  Canvas", image)
	# cv2.imshow("Backend Canvas", imageCanvas)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
