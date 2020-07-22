import concurrent.futures
import logging
import queue
import random
from threading import Thread
import time
from flask import Flask, render_template, url_for
from flask_socketio import SocketIO
import base64
import os
# Creating flask app and initialize the socket
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


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Socket
socketio = SocketIO(app, cors_allowed_origins = "*",async_mode=async_mode)

# Load the html page as homepage
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


if __name__ == "__main__":
    socketio.run(app, port = 9004)
    print ("socket listening on 9004")
