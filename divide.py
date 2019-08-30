import os
import cv2

cap=cv2.VideoCapture('outtest.avi')
for i in range(67-24):
    namei=str(i)
    os.makedirs(namei,exist_ok=True)
while True:
    ret, frame=cap.read()
    if not ret:
        break
    frame_no=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for i in range(67-24):
        namei=str(i)
        if(frame_no>=i and frame_no<i+24):
            filepath=os.path.join(namei,'frame_{:04d}.png'.format(frame_no))
        cv2.imwrite(filepath,frame)
cap.release()