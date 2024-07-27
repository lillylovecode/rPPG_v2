import cv2
import numpy as np
import time

class Webcam(object):
    def __init__(self):
        #print ("WebCamEngine init")
        self.dirname = "" #for nothing, just to make 2 inputs the same
        self.cap = None
        self.fps2=0
        self.camera = 0
        
    def start(self):
        #print("[INFO] Start webcam")
        time.sleep(1) # wait for camera to be ready

        
        self.cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW) #1為外接鏡頭，0為電腦自己鏡頭
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
        #self.cap.set(cv2.CAP_PROP_FPS, 100)
        self.fps2=self.cap.get(cv2.CAP_PROP_FPS)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
    
    def get_frame(self):
    
        if self.valid:
            #self.cap.set(cv2.CAP_PROP_FPS, 100)
            _,frame = self.cap.read()
            frame = cv2.flip(frame,1)            
            frame = cv2.resize(frame,(675,525))
            #cv2.putText(frame, str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       #(65,220), cv2.FONT_HERSHEY_PLAIN, 2, (0,256,256))
        else:
            frame = np.ones((60,80,3), dtype=np.uint8)
            #切的像素格
            col = (0,256,256)
            #cv2.putText(frame, "(Error: Camera not accessible)",
            #          (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            #print("[INFO] Stop webcam")

