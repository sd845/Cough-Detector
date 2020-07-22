from flask import Flask, render_template
from flask_socketio import SocketIO
import time

app = Flask(__name__)
socket = SocketIO(app, cors_allowed_origins="*", ping_interval=60000, ping_timeout=120000, async_mode='threading')

@socket.on('connect')
def connection():
    print ("Connected")
    socket.emit('message', 'connected established')

@socket.on('msg')
def handler(d):
    print ("Msg from client ")
    time.sleep(60)
    print ("Ack now ", d['index'])
    socket.emit('ack', d['index'])


if __name__ == "__main__":
    print ("Running on port 9002")
    socket.run(app, port=9002, log_output=False)
