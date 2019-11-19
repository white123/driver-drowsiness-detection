#!/usr/bin/env python3

import logging
import threading
import sys
import os
import json
from cv2 import cv2 as cv
from queue import Queue
from typing import Union

from flask import Flask, render_template, Response
from flask_socketio import SocketIO

from driver import DriverClass


logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

video_file = 'data/videos/IMG_6782.mp4'
#video_file = 'data/videos/incar_rgb_201908016-2.mp4'

source_type = os.environ['INPUT_SRC'] if 'INPUT_SRC' in os.environ else 'camera'
avail_source_type = ['video', 'camera']
name = ""

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app)

def detect_image(model_class: str, frame_q: Queue):
    while True:
        frame = frame_q.get()
        if frame is None: break
        info = model_class.detect_image(frame)
        socketio.emit('server_response', json.dumps(info))

def capture_frames(source: Union[str, int], frame_q: Queue, model_class):
    try:
        cap = cv.VideoCapture(source)
        while cap.isOpened():
            # Capture frame-by-frame
            retval, frame = cap.read()
            if retval:
                frame_q.put(cv.flip(frame,1))

                yield (
                    b'--frame\r\n' +
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    cv.imencode('.jpg', cv.flip(frame,1))[1].tobytes() + 
                    b'\r\n')
            cv.waitKey(1)
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():        
    return Response(
        capture_frames(0 if source_type == 'camera' else video_file, frame_q, model_class),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    frame_q = Queue(1)
    
    if source_type not in avail_source_type:
        logging.error(f'Source type {source_type} not supported')

    # list all users
    path = './data/photos'
    names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print('=' * 15, 'Driver List', '=' * 15)
    for n in names:
        if n.split('.')[-1] == 'jpg':
            print(n.split('.')[0])
    print('=' * 43, end='\n\n')

    # Init Model class
    name = input("Enter Driver's ID: ")
    model_class = DriverClass(name)

    logging.info(f'Start to detect object with model DriverClass')
    threading.Thread(target=detect_image, args=(model_class, frame_q), daemon=True).start()
    
    logging.info(f'Start to load from source input {source_type}')
    logging.info(f'Use source type {source_type}')
    
    socketio.run(app)
