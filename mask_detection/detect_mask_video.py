from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
import os
import numpy as np
def detect_and_predict_mask(frame,facenet,maskmodel):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    facenet.setInput(blob)
    detections = facenet.forward()
    faces = []
    locs=[]
    preds=[]

    for i in range(0,detections.shape[2]):
        confidense = detections[0,0,i,2]
        if confidense > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startX:endX,startY:endY]
            if face.any():
                face= cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face,(224,224))
                face= img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX,startY,endX,endY))
    if len(faces)>0:
        faces = np.array(faces)
        preds = maskModel.predict(faces)
    return (locs,preds)


prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
weightsPath = os.path.sep.join(['face_detector',"res10_300x300_ssd_iter_140000.caffemodel"])
facenet = cv2.dnn.readNet(prototxtPath,weightsPath)

maskModel = load_model('trained_model.h5')

vs=VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) =detect_and_predict_mask(frame, facenet, maskModel)
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,without_mask) = pred
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
