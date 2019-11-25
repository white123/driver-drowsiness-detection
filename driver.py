# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
import threading
import requests
from cv2 import cv2 as cv
import time
import datetime
import json
import utility
import os
import logging

from queue import Queue

from driver_age_gender import AgeGenderClass
from driver_behavior import BehaviorClass
from driver_emotion import EmotionClass
from driver_face_detection import FaceDetectionClass
from driver_identification import IdentificationClass

class DriverClass():
    def __init__(self, name):
        self.max_frame_count = int(os.getenv('MAX_FRAME_COUNT', 0))  # default = 0: no frame skipped
        self.frame_count = self.max_frame_count
        self.info = []
        
        self.padding_ratio = 0.15

        self.obj_behavior = BehaviorClass()
        self.obj_face = FaceDetectionClass()
        self.obj_identification = IdentificationClass()

        self.vote_num = 15
        self.drowsiness_vote = Queue(self.vote_num)
        self.drowsiness_sum = 0
        self.yawn_vote = Queue(self.vote_num)
        self.yawn_sum = 0
        for _ in range(0, self.vote_num):
            self.drowsiness_vote.put(False)
            self.yawn_vote.put(False)
        self.driving = False
        self.driver_name = name
        self.driver_vote = 0

        # if utility.is_recording:
        #     utility.start_record(self.image_w, self.image_h)

    def detect_models(self, image):
        
        # get faces in image
        t = time.time()
        driver_face = self.obj_face.get_driver_dlib(image)
        duration_face = time.time() - t
        # print(f'Face Detection Time = {duration_face} sec', end='   ')

        if driver_face is None:
            # logging.info("!!! No face found !!!")
            self.info = ""
            return

        # for rect in faces:
        x1, x2, y1, y2 = driver_face.left(), driver_face.right(), driver_face.top(), driver_face.bottom()
        padding_w = int((x2 - x1) * self.padding_ratio)
        padding_h = int((y2 - y1) * self.padding_ratio)
        face = image[max(0, y1 - padding_h):min(y2 + padding_h, image.shape[0] - 1), 
                    max(0, x1 - padding_w):min(x2 + padding_w, image.shape[1] - 1)]
        face = cv.resize(face, (180, 180))

        # Get name
        t = time.time()
        name = self.obj_identification.get_name(face)
        duration_name = time.time() - t
        # print('Driver Identification Time = {:8.6f} sec'.format(duration_name), end='   ')

        # Check drowsiness and yawn
        t = time.time()
        drowsiness, yawn, gaze, head_pose, theta = self.obj_behavior.check_drowsiness_yawn(image, driver_face)
        duration_yawn = time.time() - t
        # print('Drowsiness and Yawn Detection Time = {:8.6f} sec'.format(duration_yawn))


        self.info = [name, drowsiness, yawn, gaze, x1, y1, x2, y2, 
                    duration_face, duration_name, duration_yawn, head_pose, theta]

    def detect_image(self, image: np.ndarray):
        # Crop image
        
        # image = cv.resize(image, (640, 360))
        ori_h, ori_w = image.shape[:2]

        image = image[ori_h//6: ori_h*5//6, 
                    ori_w//4: ori_w*3//4]

        if self.frame_count >= self.max_frame_count:
            self.detect_models(image)
            self.frame_count = 0
        else:
            self.frame_count += 1
        
        if not self.info:
            return {}

        name, drowsiness, yawn, gaze, x1, y1, x2, y2, duration_face, \
        duration_name, duration_yawn, head_pose, theta = self.info

        duration = duration_face + duration_name + duration_yawn

        '''show driver status'''
        self.drowsiness_sum += drowsiness
        self.drowsiness_sum -= self.drowsiness_vote.get()
        self.drowsiness_vote.put(drowsiness)
        self.yawn_sum += yawn
        self.yawn_sum -= self.yawn_vote.get()
        self.yawn_vote.put(yawn)

        if int(self.yawn_sum * 1.1) + self.drowsiness_sum * 2 > self.vote_num:
            safety_status = 'Danger'
        elif int(self.yawn_sum * 1.1) + self.drowsiness_sum * 2 > self.vote_num // 2:
            safety_status = 'Warning'
        else:
            safety_status = 'Safe'

        '''show driving status'''
        if name == self.driver_name and self.driver_vote < 20:
            self.driver_vote += 1
        elif self.driver_vote > 0:
            self.driver_vote -= 1

        if not self.driving:
            driving_status = "Vehicle Locked"
            if self.driver_vote >= 15:
                self.driving = True
        else:
            driving_status = "Driving Normally"
            if self.driver_vote < 5:
                driving_status = "Invalid Driver"

        # print(f'theta: {theta}')

        return {
            'name': name, 'drowsiness': drowsiness, 'yawn': yawn, 'gaze': gaze,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'duration_face': float(duration_face), 'duration_name': float(duration_name),
            'duration_yawn': float(duration_yawn), 'duration': float(duration), 'head_pose': head_pose,
            'theta': theta, 'safety_status': safety_status, 'driving_status': driving_status
        }
