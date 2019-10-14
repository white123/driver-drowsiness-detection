# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
from cv2 import cv2 as cv
import time
import datetime
import json
import os

is_showing = True  # True: use cv.imshow(); False: use cv.imwrite()
is_showing = str(os.getenv('FLAG_IS_SHOWING', str(is_showing))).lower() == "true"
print('is_showing = {}' .format(is_showing))

is_recording = False
is_recording = (str(os.getenv('FLAG_IS_RECORDING', str(is_recording))).lower() == "true")
print('is_recording = {}' .format(is_recording))
recording = None

rect_h = 14
rect_w = 140
textsize_fps = 0.4
textsize_label = 0.4

textsize_detail = 0.3
detail_h = 12

def start_record(image_w, image_h):
    global recording

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    recording = cv.VideoWriter("output/result.mp4", fourcc, 20, (image_w, image_h))

def stop_record():
    global recording

    if recording is not None:
        recording.release()
        recording = None

def resize_and_pad(image, size_w, size_h, pad_value=114):
    image_h, image_w = image.shape[:2]

    is_based_w = float(size_h) >= (image_h * size_w / float(image_w))
    if  is_based_w:
        target_w = size_w
        target_h = int(np.round(image_h * size_w / float(image_w)))
    else:
        target_w = int(np.round(image_w * size_h / float(image_h)))
        target_h = size_h
        
    image = cv.resize(image, (target_w, target_h), 0, 0, interpolation=cv.INTER_NEAREST)
    #image = cv.resize(image, (target_w, target_h), 0, 0, interpolation=cv.INTER_LINEAR)

    top = int(max(0, np.round((size_h - target_h) / 2)))
    left = int(max(0, np.round((size_w - target_w) / 2)))
    bottom = size_h - top - target_h
    right = size_w - left - target_w
    image = cv.copyMakeBorder(image, top, bottom, left, right,
                               cv.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

    return image

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

def draw_object(image, color, label, confidence, x1, y1, x2, y2, message=None):
    cv.rectangle(image, (x1, y1), (x2, y2), color, 1)

    y = y1 - rect_h
    if y < 0:
        y = y2
    cv.rectangle(image, (x1, y), (x1 + rect_w, y + rect_h), color, -1)
    cv.putText(image, label, (x1 + 4, y + rect_h - 4), cv.FONT_HERSHEY_SIMPLEX, textsize_label, (255 - color[0], 255 - color[1], 255 - color[2]), 1, cv.LINE_AA)

    if message is None:
        message = { "Label": label,
                    "Confidence": "{:6.4f}".format(confidence),
                    "Position": [int(x1), int(y1), int(x2), int(y2)],
                    "TimeStamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }
    # print('detection result: {}' .format(json.dumps(message)))

def output_result(image, duration):
    if duration > 0.0:
        # Write detection time
        fps = 1.0 / duration
        duration = int(np.round(duration * 1000))
        text = "Detect 1 frame : {} ms | {:6.2f} fps" .format(duration, fps)
        cv.putText(image, text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, textsize_fps, (255, 255, 255), 1, cv.LINE_AA)
        #print(text)

    # Reduce image size to speed up image saving
    image_h, image_w = image.shape[:2]
    #image = cv.resize(image, (int(image_w / 2), int(image_h / 2)))

    if is_showing:
        if recording is not None:
            recording.write(image)
        #cv.imshow("Detection Result", image)
        # if (cv.waitKey(25) & 0xff) == ord('q'):
        #     if is_recording:
        #         stop_record()
    else:        
        cv.imwrite("output/result.jpg", image)
    return image

def output_detail(image, index, item, duration):
    duration = int(np.round(duration * 1000))
    text = item + ": {} ms" .format(duration)
    cv.putText(image, text, (20, 32 + detail_h * index), cv.FONT_HERSHEY_SIMPLEX, textsize_detail, (255, 255, 255), 1, cv.LINE_AA)
