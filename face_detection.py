import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils

class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=200)
        self.fps2 = 0       
    
    def face_detect(self, frame, _idxROI=3):
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        leftEye = np.zeros((6, 2), np.uint8)
        rightEye = np.zeros((6, 2), np.uint8)
        status = False
        roi = _idxROI  #選擇ROI 0:全臉 1:額頭 2:下巴 3:臉頰

        if frame is None:
            return 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects)>0:
            status = True
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            
            if y<0:
                return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
            face_frame = frame[y:y+h,x:x+w]
            
            if(face_frame.shape[:2][1] != 0):
                face_frame = imutils.resize(face_frame,width=256)
            
            face_frame = self.fa.align((frame),(gray),(rects[0])) 

            grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            rectsf = self.detector(grayf, 0)
            
            try:
                shape = self.predictor(grayf, rectsf[0])
                shape = face_utils.shape_to_np(shape)
            #計算眨眼
                leftEye = shape[lStart:lEnd] #提取人眼座標
                rightEye = shape[rStart:rEnd]
            except:
                return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
            if len(rectsf) >0:
                
                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) 
                    
                    #draw facial landmarks
                    #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                    #那幾個點的大小

                if(roi == 3):
                    #1眼睛下方(3.臉頰)
                    cv2.rectangle(face_frame,(shape[54][0], shape[29][1]), (shape[12][0],shape[33][1]), (0,255,0), 0)
                    cv2.rectangle(face_frame, (shape[4][0], shape[29][1]), (shape[48][0],shape[33][1]), (0,255,0), 0)   
                                
                    ROI1 = face_frame[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]   #right cheek
                    ROI2 = face_frame[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]    #left cheek
                    '''
                    #draw rectangle on right and left cheeks
                    #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
                    #顯示成綠色
                    '''

                elif(roi == 2):
                    #2下巴(2.下巴)
                    cv2.rectangle(face_frame,(shape[64][1], shape[67][1]),(shape[20][1],shape[6][1]), (0,255,0), 0) #框出下巴

                    ROI1 = face_frame[shape[67][1]:shape[6][1],shape[20][1]:shape[64][1]] #將下巴區域取出給ROI1
                    ROI2 = face_frame[shape[67][1]:shape[6][1],shape[20][1]:shape[64][1]] #將下巴區域取出給ROI2

                elif(roi == 0):
                    #3眼睛以下臉部區域(0.全臉)
                    cv2.rectangle(face_frame,(shape[0][0], shape[28][1]), (shape[26][0],shape[9][1]), (0,255,0), 0)

                    ROI1 = face_frame[shape[28][1]:shape[9][1], shape[0][0]:shape[26][0]]
                    ROI2 = face_frame[shape[28][1]:shape[9][1], shape[0][0]:shape[26][0]]
                
                elif(roi == 1):
                    #4額頭(1.額頭)
                    cv2.rectangle(face_frame, (shape[76][0], shape[19][1]), (shape[73][0], shape[73][1]), (0, 255, 0), 0) #框出額頭

                    #索引值的排序必須看shape裡面的值大小，必須由小到大，否則取不到訊號(反向索引)
                    ROI1 = face_frame[shape[73][1]:shape[19][1], shape[76][0]:shape[73][0]] #將額頭區域取出給ROI1
                    ROI2 = face_frame[shape[73][1]:shape[19][1], shape[76][0]:shape[73][0]] #將額頭區域取出給ROI2

                #0是x軸
                #1是y軸
                #get the shape of face for color amplification
                rshape = np.zeros_like(shape) 
                rshape = self.face_remap(shape)    
                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))
            
                cv2.fillConvexPoly(mask, rshape[0:27], 1) 
                #函數原型：void fillConvexPoly(Mat& img, const Point* pts, int npts, const Scalar& color, int lineType= 8 , int shift= 0 )
                #函數作用：填充凸多邊形
                #參數說明：img 圖像
                #pts 指向單個多邊形的指針數組
                #npts 多邊形的頂點個數
                #color 多邊形的顏色
                #LineType 組成多邊形的線條的類型
                                  #8 (or 0 ) - 8 - connected line（8鄰接)連接線。
                                  #4 - 4 - connected line(4鄰接)連接線。
                                  #CV_AA - antialiased線條。
                #shift頂點坐標的小數點位數
                #函數說明：函數fillConvexPoly填充凸多邊形內部。
                #這個函數比函數cvFillPoly更快。
                #它除了可以填充凸多邊形區域還可以填充任何的單調多邊形。
                #例如：一個被水平線（掃描線）至多兩次截斷的多邊形
                # mask = np.zeros((face_frame.shape[0], face_frame.shape[1],3),np.uint8)
                # cv2.fillConvexPoly(mask, shape, 1)
      
        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False
        return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
    
    # some points in the facial landmarks need to be re-ordered
    
    def face_remap(self,shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]
        
        remapped_image = cv2.convexHull(shape)
        return remapped_image
    
