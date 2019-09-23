# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
import onnxruntime as rt
import utility

class AgeGenderClass():
    def __init__(self):

        # Initialize emotion classfication model
        self.model_file_age = 'models/emotion_ferplus/age2.onnx'
        self.model_file_gender = 'models/emotion_ferplus/gender2.onnx'

        # size_w and size_h need to be divisible of 32 as mentioned in 
        # https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn#preprocessing-steps
        self.size_w = 224
        self.size_h = 224

        # Load model
        self.session_age = rt.InferenceSession(self.model_file_age)        
        self.inputs_age_name = self.session_age.get_inputs()[0].name

        self.session_gender = rt.InferenceSession(self.model_file_gender)        
        self.inputs_gender_name = self.session_gender.get_inputs()[0].name

    def get_age(self, face):
        # Preprocess input image for age detection
        image_data = utility.resize_and_pad(face, self.size_w, self.size_h)
        image_data = np.ascontiguousarray(np.array(image_data, dtype=np.float32).transpose(2, 0, 1)) # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0)

        # Detect age
        result = self.session_age.run(None, {self.inputs_age_name: image_data})

        return result[0][0][0]

    def get_gender(self, face):
        # Preprocess input image for gender detection
        image_data = utility.resize_and_pad(face, self.size_w, self.size_h)
        image_data = np.ascontiguousarray(np.array(image_data, dtype=np.float32).transpose(2, 0, 1)) # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0)

        # Detect gender
        result = self.session_gender.run(None, {self.inputs_gender_name: image_data})

        return result[0][0][0]
