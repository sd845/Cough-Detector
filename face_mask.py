from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import imutils
import cv2
import os
import librosa
import tensorflow
tensorflow.get_logger().setLevel('INFO')

##--------------------------------------------------------------------------------------------------------
## Mask Detection functions
##--------------------------------------------------------------------------------------------------------
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
		if confidence > 0.70:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# print("box:",(startX, startY, endX, endY))

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering,
			# resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			# print("face:",(startX, startY, endX, endY))


			try:
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				# add the face and bounding boxes to their respective lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

			except:
				return (-1,-1,-1)

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds, len(faces))



def init_face_mask():


	# load our serialized face detector model from disk
	mypath = os.getcwd()
	facenet_dir = os.path.join(mypath,"face_detector")
	prototxtPath = os.path.join(facenet_dir,"deploy.prototxt")
	weightsPath = os.path.join(facenet_dir,"res10_300x300_ssd_iter_140000.caffemodel")
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("Loaded faceNet model")

	# load the face mask detector model from disk
	json_file = open(os.path.join(mypath,"detection_model.json"),"r")
	loaded_model_json = json_file.read()
	json_file.close()
	maskNet = model_from_json(loaded_model_json)
	# load weights into new model
	maskNet.load_weights(os.path.join(mypath,"detection_model.h5"))
	print("Loaded maskNet model")
	return faceNet,maskNet


##--------------------------------------------------------------------------------------------------------
## Cough Detection functions
##--------------------------------------------------------------------------------------------------------

def init_cough_mask():
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
		return model

def extract_features(file_name):

	audio, sample_rate = librosa.load(file_name)
	mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
	mfccs = mfccs[:40,:216]

	return mfccs

def print_prediction(model,file_name):

    prediction_feature = extract_features(file_name)
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

        if (i) == labelid:
            probability = round(predicted_proba[i],2)

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
