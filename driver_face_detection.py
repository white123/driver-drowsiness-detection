# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
from cv2 import cv2 as cv
import dlib

class FaceDetectionClass():
    def __init__(self):

        # Load OpenCV pretrained Haar-cascade face classifier
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#haar-cascade-detection-in-opencv
        #self.face_classifier_file = 'models/emotion_ferplus//haarcascade_frontalface_default.xml'
        #self.face_cascade = cv.CascadeClassifier(self.face_classifier_file)

        # Initialize face detection model
        #self.face_proto = "models/emotion_ferplus/opencv_face_detector.pbtxt"
        #self.face_model = "models/emotion_ferplus/opencv_face_detector_uint8.pb"
        #self.face_net = cv.dnn.readNet(self.face_model, self.face_proto)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/emotion_ferplus/shape_predictor_68_face_landmarks.dat")

    def get_faces_cv(self, net, image, conf_threshold=0.6):
        image_h, image_w = image.shape[:2]
        blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
            
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * image_w)
                y1 = int(detections[0, 0, i, 4] * image_h)
                x2 = int(detections[0, 0, i, 5] * image_w)
                y2 = int(detections[0, 0, i, 6] * image_h)
                faces.append([x1, y1, x2, y2])
        return faces
    
    def get_driver_dlib(self, image):
        try:
            # get faces in image
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # gray image

            #faces = self.get_faces(self.face_net, image)
            #faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            #faces = self.detector(gray, 1)
            img = image.copy()
            img = cv.filter2D(img, -1, kernel=np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], np.float32))

            # Convert to RGB image compatible to dlib.load_rgb_image(f)
            # http://dlib.net/face_landmark_detection.py.html
            img = img[:, :, [2, 1, 0]]  # BGR => RGB
            
            #faces = self.detector(img, 1)
            faces = self.detector(img)

        except Exception as ex:
            print("Exception in detect_image: %s" % ex)
            faces = None

        face = None
        area = 0
        for i in faces:
            face_area = (i.right() - i.left() + 1) * (i.bottom() - i.top() + 1)
            if face_area > area:
                face = i
                area = face_area
        
        return face
