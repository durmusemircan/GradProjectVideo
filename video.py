import tensorflow as tf
import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from video_app import  Ui_MainWindow

class video(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.liveCam.setScene(self.scene)
        self.path = "haarcascade_frontalface_default.xml"
        self.model = tf.keras.models.load_model('GradProjectVideoModel.h5')
        self.cam = cv2.VideoCapture(0)

        self.time = QTimer()
        self.time.timeout.connect(self.framing)
        self.time.start(10)

        self.time2predict = QTimer()
        self.time2predict.timeout.connect(self.predicting)
        self.time2predict.start(2000)

        self.face = []


        
    def framing(self):
        ret,frame = self.cam.read()
        if ret:
            self.newFrame = frame
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.face = faceCascade.detectMultiScale(gray, 1.1, 4)
            for x,y,w,h in self.face:               
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            self.liveCamShow(frame)

    def predicting(self):        
        if len(self.face) == 0:
            print("FACE COULD NOT FOUND")
        else:
            for (x,y,w,h) in self.face:
                face_roi = self.newFrame[y: y+h, x:x + w] 
                
            final_img = cv2.resize(face_roi, (224,224))
            final_img = np.expand_dims(final_img, axis = 0)
            final_img = final_img/255.0

            Prediction = self.model.predict(final_img)

            if(np.argmax(Prediction)==0):
                status = "ANGRY"

            elif(np.argmax(Prediction)==1):
                status = "DISGUST"

            elif(np.argmax(Prediction)==2):
                status = "FEAR"

            elif(np.argmax(Prediction)==3):
                status = "HAPPY"

            elif(np.argmax(Prediction)==4):
                status = "SAD"

            elif(np.argmax(Prediction)==5):
                status = "SURPRISE"

            else:
                status = "NEUTRAL"

            self.progressBar_Angry.setValue(int(Prediction[0][0] * 100))
            self.progressBar_Disgust.setValue(int(Prediction[0][1] * 100))
            self.progressBar_Fear.setValue(int(Prediction[0][2] * 100))
            self.progressBar_Happy.setValue(int(Prediction[0][3] * 100))
            self.progressBar_Neutral.setValue(int(Prediction[0][6] * 100))
            self.progressBar_Sad.setValue(int(Prediction[0][4] * 100))
            self.progressBar_Surprise.setValue(int(Prediction[0][5] * 100))
            self.statusText.setPlainText(status)

    def liveCamShow(self, frame):
        img = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(img)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.liveCam.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = video()
    window.show()
    sys.exit(app.exec_())