# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
from cv2 import cv2 as cv
import face_recognition
import os
import requests
import base64

class IdentificationClass():
    def __init__(self):
        # Initialize face recognition model
        self.tmpfile = "output/img.jpg"
        self.known_face_encodings = []
        self.known_face_names = []
        self.update_known_list()
        self.face_threadhold = 0.45

    def update_known_list(self):

        # get photo from api
        # url = "https://webappdriverenrollment.azurewebsites.net/api/Drivers/All"
        # headers = {'Content-Type': 'application/octet-stream'}
        # try:
        #     response = requests.get(url, headers=headers)
        # except Exception as ex:
        #     print("Exception in requests.get(): %s" % ex)
        #     return

        # if response.status_code != requests.codes['ok']:
        #     return

        # drivers = response.json()


        # get photo from dir
        drivers = []
        photosPath = './data/photos/'
        for file in os.listdir(photosPath):
            with open(photosPath+file, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            drivers.append({'Name':file[:-4], 'PhotoFile':file, 'Photo': encoded_string})

        known_face_encodings = []
        known_face_names = []
        for driver in drivers:
            try:            
                bytes = base64.b64decode(driver['Photo'])
                np_arr = np.fromstring(bytes, np.uint8)
                img = cv.imdecode(np_arr, -1)
                cv.imwrite(self.tmpfile, img)
                known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(self.tmpfile))[0])
                known_face_names.append(driver['Name'])
            except Exception as ex:
                print("Exception in building driver known list: %s" % ex)
        self.known_face_encodings.clear()
        self.known_face_encodings = known_face_encodings
        self.known_face_names.clear()
        self.known_face_names = known_face_names
        
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)

    def get_name(self, img):
        if len(self.known_face_names) == 0 or len(self.known_face_encodings) == 0:
            return "Unknown"

        # Find all the faces and face enqcodings in the frame of video
        face = img[:, :, [2, 1, 0]]  # BGR => RGB
        face_locations = face_recognition.face_locations(face)
        face_encodings = face_recognition.face_encodings(face, face_locations)
        name = "Unknown"
        # Loop through each face in this frame of video
        for (_, _, _, _), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            min_dis = face_distances.min()
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and min_dis < self.face_threadhold:
                name = self.known_face_names[best_match_index]
            
            if (name != "Unknown"):
                break
        
        #print('name = {}' .format(name))
        return name
