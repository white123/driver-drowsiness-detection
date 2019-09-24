import cv2
import time

#cap = cv2.VideoCapture('data/videos/IMG_6793.mp4')
cap = cv2.VideoCapture('data/videos/I.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('total frames:', total)

t = time.time()
cnt = 1
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (320, 180))
    cv2.imshow('t', frame)
    cv2.waitKey(1)
    duration = time.time() - t
    print('\b'*20, end='')
    print(cnt, end='', flush=True)
    cnt += 1
