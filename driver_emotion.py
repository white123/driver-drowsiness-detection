# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import numpy as np
import onnxruntime as rt
import utility

class EmotionClass():
    def __init__(self):

        # Initialize emotion classfication model
        self.model_file = 'models/emotion_ferplus/model.onnx'
        self.threshold = 0.5
        self.padding_ratio = 0.15
        self.labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
        self.colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),(0,0,64),(0,64,0)]

        # size_w and size_h need to be divisible of 32 as mentioned in 
        # https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn#preprocessing-steps
        self.size_w = 64
        self.size_h = 64        
        self.input_shape = (1, 1, 64, 64)

        # Load model
        self.session = rt.InferenceSession(self.model_file)        
        self.inputs = self.session.get_inputs()
        for i in range(len(self.inputs)):
            print("input[{}] name = {}, type = {}" .format(i, self.inputs[i].name, self.inputs[i].type))

    def get_emotion(self, face_gray):
        # Preprocess input image for emotion detection
        image_data = utility.resize_and_pad(face_gray, self.size_w, self.size_h, 0)
        image_data = np.array(image_data, dtype=np.float32)
        image_data = np.resize(image_data, self.input_shape)

        # Detect emotion
        result = self.session.run(None, {self.inputs[0].name: image_data})

        # Postprocess output data and draw emotion label
        scores = result[0][0]
        for i in range(len(scores)):
            scores[i] = max(scores[i], 1e-9)   # convert negative value to be 1e-9
        scores = utility.softmax(scores)
        class_index = np.argmax(scores)
        confidence = scores[class_index]
        color = self.colors[class_index]
        emotion = self.labels[class_index]
        return emotion, confidence, color
