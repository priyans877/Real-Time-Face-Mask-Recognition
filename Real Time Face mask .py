# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_predict(frame, netface, maskNet):
	# dimensionsoning of the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	
	netface.setInput(blob)
	detecting = netface.forward()
	print(detecting.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detecting.shape[2]):
		# extract the Possibility associated with image
		Possibility = detecting[0, 0, i, 2]

		if Possibility > 0.5:

			# compute the (x, y)-coordinates of the bounding box for
			box = detecting[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
	
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make predictions if at least one face was detected
	if len(faces) > 0:
		# for faster performence checking one face at one time
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

# load face detector model
prototxtPath = r"ourmodels\deploy.prototxt"
weightsPath = r"ourmodels\res10_300x300_ssd_iter_140000.caffemodel"
netface = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading mask detector model
maskNet = load_model("mask_detector.model")

# video stream
print("..Starting live detection...")
vs = VideoStream(src=0).start()

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_predict(frame, netface, maskNet)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
		# display detection area
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()