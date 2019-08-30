import cv2
import numpy as np
import os
from PIL import Image

def grey(image):
    greyarray= image[:,:,0]* 0.2989 + image[:,:,1]*0.5870  + image[:,:,2]*0.1140
    #img = Image.fromarray(greyarray)
    #img.show()
    return greyarray.reshape(1,-1)

cap = cv2.VideoCapture(1)
os.makedirs('imagefolder',exist_ok=True)
os.makedirs('videofolder',exist_ok=True)
os.makedirs('datafolder',exist_ok=True)
#保存
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 20.0
size = (120, 90)
number=20
for i in range(number):
    writer = cv2.VideoWriter(os.path.join('videofolder','video{:02d}.avi'.format(i)), fmt, fps, size)
    j=0
    while j<60 :
        _, frame = cap.read()
        frame = cv2.resize(frame, size)
        #保存
        writer.write(frame)
        cv2.imshow('frame', frame)
        j+=1
    writer.release()
    cv2.destroyAllWindows()
cap.release()

for i in range(number):
    cap=cv2.VideoCapture(os.path.join('videofolder','video{:02d}.avi'.format(i)))
    os.makedirs(os.path.join('imagefolder','output{:02d}'.format(i)),exist_ok=True)
    for j in range(60):
        ret, frame=cap.read()
        if not ret:
            break
        frame_no=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        filepath=os.path.join('imagefolder','output{:02d}'.format(i),'frame_{:04d}.png'.format(frame_no))
        cv2.imwrite(filepath,frame)
    cap.release()

for i in range(number):
    file= open(os.path.join('datafolder',"picdata{:02d}.txt".format(i)),"w+")
    for j in range(60):
        image=Image.open(os.path.join('imagefolder','output{:02d}'.format(i),'frame_{:04d}.png'.format(j+1)))
        greyarray=grey(np.array(image)).astype(int)#230400*1
        np.savetxt(file, greyarray, fmt="%d")#60*230400
    file.close()
