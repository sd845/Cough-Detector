from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations, and the list of predictions
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.50:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			try:
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))
			except:
				pass

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

def init_face_mask():
	# load our serialized face detector model from disk
	mypath = os.getcwd()
	facenet_dir = os.path.join(mypath,"face_detector")
	prototxtPath = os.path.join(facenet_dir,"deploy.prototxt")
	weightsPath = os.path.join(facenet_dir,"res10_300x300_ssd_iter_140000.caffemodel")
	#prototxtPath = r'C:\\Users\\Mitali Sheth\\Documents\\Anaconda Navigator Files\\Pnuemosense\\Cough-Detector\\face_detector\\deploy.prototxt'
	#weightsPath = r'C:\\Users\\Mitali Sheth\\Documents\\Anaconda Navigator Files\\Pnuemosense\\Cough-Detector\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel'
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("loaded face detector model...")

	# load the face mask detector model from disk
	# maskNet = load_model('/Users/srutidammalapati/face-mask-detector/mask_detector.model')
	json_file = open(os.path.join(mypath,"detection_model.json"),"r")
	#json_file = open(r'C:\\Users\\Mitali Sheth\\Documents\\Anaconda Navigator Files\\Pnuemosense\\detection_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	maskNet = model_from_json(loaded_model_json)
	# load weights into new model
	maskNet.load_weights(os.path.join(mypath,"detection_model.h5"))
	#maskNet.load_weights(r"C:\\Users\\Mitali Sheth\\Documents\\Anaconda Navigator Files\\Pnuemosense\\detection_model.h5")
	print("Loaded face model from disk") 
	return faceNet,maskNet

def check_mask(lname,filepath,faceNet,maskNet,imagepath):
	count = 1
	print("Inside check_mask: ",filepath)
	if lname == "coughing":
		is_cough = True
	else:
		is_cough = False

	
	vidcap = cv2.VideoCapture(filepath)
	#vidcap = cv2.VideoCapture(r'C:\\Users\\Mitali Sheth\\Documents\\Anaconda Navigator Files\\Pnuemosense\\audios\\7.webm')
	#success,image = vidcap.read()
	# loop over the frames from the video stream
	#print("In check_mask: ",success)
	
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*4000))
	success,frame = vidcap.read()
	print("If success: ",success)
	#if (not success):
	#	break
	print("If success: ",frame.shape)
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		
		#sort into risk category
		if label == "Mask":
			if is_cough:
				label = "Moderate Risk"
				color = (255, 0, 0) #Blue
			else:
				label = "Low Risk"
				color = (0, 225, 0) #Green
		else:
			label = "High Risk"
			color = (0, 0, 225) #Red

		# include the probability in the label
		label = "{} ({:.1f}%)".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10000) & 0xFF
	count = count + 1
	cv2.imwrite(imagepath,frame)
	return frame
		
	# do a bit of cleanup
	cv2.destroyAllWindows()