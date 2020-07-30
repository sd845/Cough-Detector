# Importing major libraries
from flask import Flask, render_template, url_for,request, jsonify
from flask_socketio  import SocketIO
import base64
import os
# Importing from external python file
from face_mask import init_cough_mask,print_prediction
from check_mask import Check_Mask
from capture import Capture
from check_mask import mask_count,re_init
#Remove logging

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#Hide warnings
import warnings
warnings.filterwarnings("ignore")
import shutil

def page_not_found(e):
    print('Not found')
    return render_template('404.html'), 404

# Creating flask app and initialize the socket
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.register_error_handler(404, page_not_found)

# Socket
socket = SocketIO(app, cors_allowed_origins="*",ping_interval=60000, ping_timeout=120000, async_mode = "threading")

#Global Variables
count = "0"
icount = "0"
lname = "breathing"
dirpath =""
Hrisk = 0
Mrisk = 0
Lrisk = 0
#Load the models
cough_model = init_cough_mask()

capture = Capture(Check_Mask())



def re_initialise():
    global Hrisk,Mrisk,Lrisk
    Hrisk = 0
    Mrisk = 0
    Lrisk = 0

# Load the html page as homepage
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

"""# Load the html page as homepage
@app.route('/robots933456.txt', methods=['GET'])
def reply():
    # Main page
    return """

# Return message on successful connection
@socket.on('connect')
def connection():
    currentSocketId = request.sid
    print ("Connected with: ",currentSocketId)
    socket.emit('message', 'connected established')

@socket.on('started')
def started_feed(message):
    print(message)
    re_initialise()
    global dirpath
    currentSocketId = request.sid
    folder = os.path.join(os.getcwd(),"static")
    dirpath = os.path.join(folder,currentSocketId)
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    print(dirpath)


#Receive the blob and process it
@socket.on('blob_event')
def handle_blob(message):
    global count
    global lname
    global dirpath
    currentSocketId = request.sid

    #Declaring filename and videoname)
    filename = "video" + count + ".webm"


    filepath = os.path.join(dirpath,filename)


    data = base64.b64decode(message)

    #write the videofile to disk
    try:
        f = open(filepath, 'wb')
        f.write(data)
        f.close()
    except:
        # print("Stoppped feed")
        return

    #Updating information
    count = str(int(count)+1)
    print ("Processing now: ", filepath)

    #function for audio processing

    lname,lprob = print_prediction(cough_model,filepath)
    print("Predicted class:",lname)
    socket.emit('label_event',{"label":lname,"prob":str(lprob)})
    # os.remove(filepath)


@socket.on('input_image')
def test_message(image):
    global icount,lname,dirpath,Hrisk,Lrisk,Mrisk

    # Saving image
    image = image.split(",")[1]
    iname = icount + ".jpg"
    currentSocketId = request.sid
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    # Creating folder to save images
    ipath = os.path.join(dirpath,"images")

    if not os.path.isdir(ipath):
        os.mkdir(ipath)
    ipath = os.path.join(ipath,iname)
    # Saving image
    with open(ipath, "wb") as f:
        f.write(base64.b64decode(image))

    # Send image for processing
    capture.enqueue_input(ipath,lname)
    icount = str(int(icount)+1)

    imagepath = os.path.join(currentSocketId,"images")
    imagepath = os.path.join(imagepath,iname)

    frame = capture.get_frame()

    if os.sep == '\\':
        imagepath = imagepath.replace('\\','/')

    # Response with the processed image
    # print("Sending image")
    socket.emit("response_back",url_for("static",filename = imagepath))
    if int(icount) % 5 ==0:
        hrisk,mrisk,lrisk = mask_count()
        re_init()
        Hrisk += hrisk
        Lrisk += lrisk
        Mrisk += mrisk
        print("High Risk: ",Hrisk,"Moderate Risk: ",Mrisk,"Low Risk: ",Lrisk)
        socket.emit('stopped',{"High Risk": Hrisk,"Moderate Risk":Mrisk,"Low Risk":Lrisk})

@socket.on('stopped')
def stopped_function(message):
    print(message)
    global dirpath
    shutil.rmtree(dirpath)
    print("In Stop: ",dirpath)

@socket.on('disconnect')
def disconnect_function():
    try:
        global dirpath
        shutil.rmtree(dirpath)
    except:
        pass
    print("Disconnected the socket: ")
    socket.emit("Manual Disconnect")

@socket.on('uncaughtException')
def exceptiony():
    # handle or ignore error
    console.log(exception);


if __name__ == '__main__':
    print ("socket listening on 8000")
    socket.run(app, port = 8000, host='0.0.0.0')
