# Driver Drowsiness Detection
An Advanced Vehicle Security & Safety System

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Architecture](#Architecture)
- [Face Detection & Recognition](#Face-Detection-&-Recognition)
- [Drowsiness Detection](#Drowsiness-Detection)
- [Awareness Detection](#Awareness-Detection)
- [Demo](#Demo)

## Install

```sh
$ pip3 install -r requirements.txt
```

## Usage

```sh
$ python3 main.py
```

The result will display on localhost:5000. You can simply open the page by your browser.

## Architecture

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/architecture.png)

## Face Detection & Recognition

- Face Detection
    - Crop potential driver face position to reduce latency
    - Dlib - get_frontal_face detector
    - Return bounding box
- Face Recognition
    - Python face_recognition library by Ageitgey
    - accuracy 99.3% on Labeled Faces in the Wild benchmark
    - Continuous vote

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/face.png)

## Drowsiness Detection

- Face Landmark Detection
    - Dlib detects 68 facial landmarks
    - Ensemble of regression trees
- Eye Closure - Eye Aspect Ratio
- Yawn

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/eye.png)

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/yawn.png)

## Awareness Detection

- Headpose estimation
- Landmark - nose, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
- SolvePnP => rotation vector
- Rodrigues => rotation matrix
- Rotation matrix => yaw, pitch


![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/awareness1.png)

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/awareness2.png)

## Demo

https://youtu.be/Wiy1606aj3w

![image](https://github.com/white123/driver-drowsiness-detection/blob/master/pic/demo.png)


