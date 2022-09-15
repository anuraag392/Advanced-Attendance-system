import face_recognition as  fr
import numpy as np
import cv2
import os
from datetime import datetime
path='images'
images=[]
personname=[]
mylist = os.listdir(path)
for cu_img in mylist:
    current_image=cv2.imread(f'{path}/{cu_img}')
    images.append(current_image)
    personname.append(os.path.splitext(cu_img)[0])
print(personname)

def faceencodings(images):
    
    encodelist=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= fr.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodelistknown=faceencodings(images)
print("All encodings completed !!!")

def attendance(name):
    with open('attendance.csv', 'w+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
       # if name not in namelist:
        time_now=datetime.now()
        tstr=time_now.strftime('%H:%M:%S')
        dstr=time_now.strftime('%d/%m/%Y')
        f.writelines(f'{name},{tstr},{dstr} \n')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None,0.25,0.25)
    faces=cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    
    facescurrentframe=fr.face_locations(faces)
    encodecurrentframe=fr.face_encodings(faces , facescurrentframe)
    
    for encodeface, faceloc in zip(encodecurrentframe , facescurrentframe):
        matches=fr.compare_faces(encodelistknown, encodeface)
        facedis=fr.face_distance(encodelistknown, encodeface)
        
        matchindex=np.argmin(facedis)
        if matches[matchindex]:
            name= personname[matchindex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1 ,(0,0,255), 2)
            attendance(name)
        
        cv2.imshow("camera", frame)
        if cv2.waitKey(10)== 13:
            break
cap.release()
cv2.destroyAllWindows()


    
    