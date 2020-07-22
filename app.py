# Importing major libraries
from flask import Flask, render_template, url_for,request
from flask_socketio  import SocketIO
import base64
import os
# Importing from external python file
from face_mask import init_face_mask,check_mask

#Threading
from threading import Thread
import time

#Remove logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#Hide warnings
import warnings
warnings.filterwarnings("ignore")

#Threading fundamentals
async_mode = None

if async_mode is None:
    try:
        import eventlet
        async_mode = 'eventlet'
    except ImportError:
        pass

    if async_mode is None:
        try:
            from gevent import monkey
            async_mode = 'gevent'
        except ImportError:
            pass

    if async_mode is None:
        async_mode = 'threading'

    print('async_mode is ' + async_mode)

# Creating flask app and initialize the socket
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Socket
socket = SocketIO(app, cors_allowed_origins="*", ping_interval=60000, ping_timeout=120000, async_mode='threading')

#Variable to create filenames
count = "0"

#Load the models
cough_model,faceNet,maskNet = init_face_mask()

# Load the html page as homepage
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# Return message on successful connection
@socket.on('connect')
def connection():
    currentSocketId = request.sid
    print ("Connected with: ",currentSocketId)
    socket.emit('message', 'connected established')


#Receive the blob and process it
@socket.on('blob event')
def handle_blob(message):
    global count
    currentSocketId = request.sid
    print('Got message from frontend :',message[:10])
    print("socket id: ",currentSocketId)

    #Declaring filename and videoname
    filename = "video" + count + ".webm"
    videoname = "vid" + count + ".avi"
    folder = os.path.join(os.getcwd(),"static")
    dirpath = os.path.join(folder,currentSocketId)
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    filepath = os.path.join(dirpath,filename)
    videopath = os.path.join(dirpath,videoname)

    #decode the message
    bstring = message
    data = base64.b64decode(bstring)

    #write the videofile to disk
    f = open(filepath, 'wb')
    f.write(data)
    f.close()

    #Updating information

    count = str(int(count)+1)
    print ("Processing now: ", videopath)

    #function for audio and video processing
    frames = check_mask(filepath,dirpath,cough_model,faceNet,maskNet,videopath)
    os.remove(filepath)
    #print ("Sending rm3 now ");
    socket.emit("rmessage3",{"message":url_for('static',filename = currentSocketId+"/"+videoname)})
    print("sent video for: ",filename)


if __name__ == '__main__':
    print ("socket listening on 9000")
    socket.run(app, port = 9000)
