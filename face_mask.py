from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import imutils
import time
import cv2
import os
import librosa

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

		# filter out weak detections
		if confidence > 0.50:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering,
			# resize it to 224x224, and preprocess it

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
				 #print("cvt error")
				 return (-1,-1)
		# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

def init_face_mask():
	#load cough detector model

	modelfolder = os.path.join(os.getcwd(),"model_6b")
	modelfjson = os.path.join(modelfolder,"model_6b.json")
	json_file = open(modelfjson, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(os.path.join(modelfolder,"model_6b.h5"))
	print("Loaded cough model from disk")

	# load our serialized face detector model from disk
	mypath = os.getcwd()
	facenet_dir = os.path.join(mypath,"face_detector")
	prototxtPath = os.path.join(facenet_dir,"deploy.prototxt")
	weightsPath = os.path.join(facenet_dir,"res10_300x300_ssd_iter_140000.caffemodel")
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("Loaded facenet model...")

	# load the face mask detector model from disk
	json_file = open(os.path.join(mypath,"detection_model.json"),"r")
	loaded_model_json = json_file.read()
	json_file.close()
	maskNet = model_from_json(loaded_model_json)
	# load weights into new model
	maskNet.load_weights(os.path.join(mypath,"detection_model.h5"))
	print("Loaded masknet model")
	return model,faceNet,maskNet

def check_mask(filepath,cough_model,faceNet,maskNet,videopath):

	count = 0
	lname,lprob = print_prediction(cough_model,filepath)
	print("Predicted class:",lname)
	if lname == "coughing":
		is_cough = True
	else:
		is_cough = False


	vidcap = cv2.VideoCapture(filepath)
	#vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000))    # added this line
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	allframes = []
	allnames = []
	success,image = vidcap.read()
	print(success)
	if success:
		print(image.shape)
	#success = True
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	#print(fps)
	video = cv2.VideoWriter(videopath, fourcc, 15.0, (640,480),True)
	#print("At present: ",os.getcwd())
	while success:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		#vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000))

		success,frame = vidcap.read()

		if (not success):
			break
		#print(frame.shape)
		#frame = imutils.resize(frame, width=400)
		height, width, layers = frame.shape

		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		if (locs == -1) and (preds == -1):
			continue;
			print("Not continued")
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
		allframes.append(frame)
		# show the output frame
		#cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		imagename = str(count)+".jpg"
		imagepath = "./images/"+ imagename
		allnames.append(imagename)
		cv2.imwrite(imagepath,frame)
		count = count + 1
		#socketio.emit("message3",url_for('static',filename = imagename))
	for f in allframes:
		video.write(f)
	video.release()
	# do a bit of cleanup
	cv2.destroyAllWindows()
	return allnames


##Cough Detection functions

def extract_features(file_name):

    audio, sample_rate = librosa.load(file_name)
    print("Audio File Loaded")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = mfccs[:40,:216]

    return mfccs

def print_prediction(model,file_name):

    prediction_feature = extract_features(file_name)
    print("Features Extracted")
    prediction_feature = librosa.util.fix_length(prediction_feature, 216)
    prediction_feature = prediction_feature.reshape(1, 40, 216, 1)
    predicted_vector = model.predict_classes(prediction_feature)



    labelid = np.int16(predicted_vector[0]).item()
    labelname = getLabel(labelid)

    #Predict probability
    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = getLabel(i)
        #print(category, "\t\t : ", format(predicted_proba[i], '.32f') )
        if (i) == labelid:
            probability = round(predicted_proba[i],2)
            #print(category," : ",probability)

    return labelname,probability


def getLabel(labelid):

    labelname = ""
    labels = {'airplane': 0, 'breathing': 1, 'car_horn': 2, 'cat': 3, 'chainsaw': 4, 'chirping_birds': 5, 'church_bells': 6,
    'clapping': 7, 'clock_alarm': 8, 'coughing': 9, 'cow': 10, 'crow': 11, 'crying_baby': 12, 'dog': 13,
    'door_wood_knock': 14, 'engine': 15, 'fireworks': 16, 'helicopter': 17, 'laughing': 18, 'laughter': 19,
    'rain': 20, 'silence': 21, 'siren': 22, 'speech': 23, 'thunderstorm': 24, 'train': 25, 'wind': 26}

    for name,i in labels.items():
        if i == labelid:
            labelname = name

    return labelname
