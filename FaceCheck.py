import cv2
import sqlite3
import sys
import os
import numpy as np
from PIL import Image

path='dataSet'
imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainningData.yml')

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203,23,252)


#get data from sqlite by ID
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        #split to get ID of the image
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return IDs, faces

def createTable(TblName):
    conn=sqlite3.connect("test.db")
    cmd='Create Table  '+ str(TblName) + '(Id  int Not null , Name Nvarchar(255), Age int, Gender Nvarchar(255) , primary key (id))'
    conn.execute(cmd)
#insert/update data to sqlite
def insertOrUpdate(id,strname,age,gender):
    conn=sqlite3.connect("test.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd='UPDATE People SET Name={}, Age ={}, Gender={} WHERE ID={}'.format('"'+str(strname) +'"',
        '"'+str(age)+'"', '"'+str(gender)+'"',str(id))
    else:
        cmd='INSERT INTO People(Id,Name) Values({},{})'.format('"'+str(strname) +'"',str(id))
    conn.close()
    conn=sqlite3.connect("test.db")
    conn.execute(cmd)
    conn.commit()
    conn.close()

cam = cv2.VideoCapture(0)
id=1
# createTable('People')
insertOrUpdate(id,'Chien Nguyen',31,'Nam')
insertOrUpdate(2,'Tony Tran',39,'Nam')
# sampleNum=0
# while(True):
#     #camera read
#     ret, img = cam.read()
#     if ret==True:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.3, 5)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
#             #incrementing sample number 
#             sampleNum=sampleNum+1
#             #saving the captured face in the dataset folder
#             cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpeg", gray[y:y+h,x:x+w])

#             cv2.imshow('frame',img)
#         #wait for 100 miliseconds 
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break
#         # break if the sample number is morethan 20
#         elif sampleNum>20:
#             break
#     cam.release()
#     cv2.destroyAllWindows()

# Ids,faces=getImagesAndLabels(path)
# #trainning
# recognizer.train(faces,np.array(Ids))
# recognizer.save('recognizer/trainningData.yml')
# cv2.destroyAllWindows()