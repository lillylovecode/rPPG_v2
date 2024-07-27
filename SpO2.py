import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from scipy import signal
import time
from face_detection import FaceDetection
from sklearn.decomposition import FastICA
#from process import Process

class SpO2_calculation(object):
    def __init__(self):
        self.allPeakTimeData=[]
        self.SpO2=0
        self.start_time_status=0
        self.fd = FaceDetection()
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.allPeakTimeData_r = []
        self.allPeakTimeData_b = []
        self.allVallyTimeData_r = []
        self.allVallyTimeData_b = []
        self.r1=0
        self.r2=0
        self.b1=0
        self.b2=0

        
        
    def closest(self,mylist, Number):
        answer = []
        for i in mylist:    
            answer.append(abs(Number-i))
        return answer.index(min(answer))

    def Z_ScoreNormalization(self,x,mu,sigma):
        x = (x - mu) / sigma;
        return x;
    
    def smoothTriangle(self,data, degree=25):
        triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
        smoothed=[]
    
        for i in range(degree, len(data) - degree * 2):
            point=data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed       
    
    def get_derivative1(self,x,maxiter=10):   #//默認使用牛頓法迭代10次
        h=0.0001
        #f=lambda x: x**2-c
        #f=lambda x: x**2-2*x-4
        F=lambda x: 0.5*x**2*(4-x)
        def df(x,f=F):                      #//使用導數定義法求解導數
                return (f(x+h)-f(x))/h

        for i in range(maxiter):
                x=x-F(x)/df(x)   #//計算一階導數，即是求解  f(x)=0  
                #x=x-df(x)/df(x,df) # //計算二階導數，即是求解  f'(x)=0
                #print (i+1,x)
        return x
    
    def get_derivative2(self,x,maxiter=10):   #//默認使用牛頓法迭代10次
        h=0.0001    
        #f=lambda x: x**2-c
        #f=lambda x: x**2-2*x-4
        F=lambda x: 0.5*x**2*(4-x)
        def df(x,f=F):                      #//使用導數定義法求解導數
                return (f(x+h)-f(x))/h

        for i in range(maxiter):
                #x=x-F(x)/df(x)   #//計算一階導數，即是求解  f(x)=0  
                x=x-df(x)/df(x,df) # //計算二階導數，即是求解  f'(x)=0
                #print (i+1,x)
        return x

    def ICA(self,norm):
        X=(norm,[1*random.random() for i in range(len(norm))])
        ICA = FastICA(n_components=1,max_iter=1000,tol=1e-6)
        X_transformed = ICA.fit_transform(X)
        A_ =  ICA.mixing_.T

        return A_ 
        
    def extractColor(self, frame):
        b = np.mean(frame[:,:,0]) # 藍色通道的平均值
        g = np.mean(frame[:,:,1]) # 綠色通道的平均值
        r = np.mean(frame[:,:,2]) # 紅色通道的平均值
        ppg=(r+g+b)/3

        return g,r,b,ppg
    
    global g1,r1,b1,ppg1 
    global g2,r2,b2,ppg2
    global r3,b3
        
    def run(self,arrRed,arrBlue,t1):
        if self.start_time_status==0:
            self.start_time=time.time()
            self.start_time_status=1
        frame, face_frame, ROI1, ROI2, status, mask , leftEye, rightEye = self.fd.face_detect(self.frame_in)
        self.frame_out = frame
        self.frame_ROI = face_frame       

        g1,r1,b1,ppg1 = self.extractColor(ROI1)
        g2,r2,b2,ppg2 = self.extractColor(ROI2)
    
        SpO2_bool = False
        SpO2 = 0
        R=0
        r3 =(r1+r2)
        b3 =(b1+b2)
    
        # allPeakTimeDataLength = 100 #這個值決定要用多少長度的allPeakTimeData[]計算SDNN，可以調整
        # first_peak, first_peak_bool = self.numPeakFun(bp,t1)
        # if(first_peak_bool==True):
        #     self.allPeakTimeData.append(first_peak)                
        #     if(len(self.allPeakTimeData)>=allPeakTimeDataLength):
        #         currentPeakTimeData = self.allPeakTimeData[-allPeakTimeDataLength:]
               
        #         r3_array = self.ac_dc_cal(currentPeakTimeData)
        #         b3_array = self.ac_dc_cal(currentPeakTimeData)
        #         SpO2 = self.SpO2_fun(r3_array, b3_array)
               
        #         self.allPeakTimeData = currentPeakTimeData
        #         SpO2_bool = True
        
        allPeakTimeDataLength = 100 #這個值決定要用多少長度的allPeakTimeData[]計算SDNN，可以調整
        first_peak_r, first_peak_bool_r = self.numPeakFun(arrRed,t1)
        first_peak_b, first_peak_bool_b = self.numPeakFun(arrBlue,t1)
        
        

        first_valley_r, first_valley_bool_r = self.numVallyFun(arrRed)
        first_valley_b, first_valley_bool_b = self.numVallyFun(arrBlue)

        if(first_peak_bool_r==True):
            self.allPeakTimeData_r.append(first_peak_r)
        if(first_peak_bool_b==True):
            self.allPeakTimeData_b.append(first_peak_b)

        if(first_valley_bool_r==True):
            self.allVallyTimeData_r.append(first_valley_r)
        if(first_valley_bool_b==True):
            self.allVallyTimeData_b.append(first_valley_b)


        if(len(self.allPeakTimeData_r)>=allPeakTimeDataLength and len(self.allPeakTimeData_b)>=allPeakTimeDataLength):
            #取最新的資料
            currentPeakTimeData_r = self.allPeakTimeData_r[-allPeakTimeDataLength:]
            currentPeakTimeData_b = self.allPeakTimeData_b[-allPeakTimeDataLength:]

            currentVallyTimeData_r = self.allVallyTimeData_r[-allPeakTimeDataLength:]
            currentVallyTimeData_b = self.allVallyTimeData_b[-allPeakTimeDataLength:]

            r3_array = self.ac_dc_cal(currentPeakTimeData_r,currentVallyTimeData_r)
            b3_array = self.ac_dc_cal(currentPeakTimeData_b,currentVallyTimeData_b)
            SpO2,R = self.SpO2_fun(r3_array, b3_array)
            SpO2_bool = True
                
        return SpO2, SpO2_bool, R
    
    def SpO2_fun(self,r3_array, b3_array):
        
        red_ac = (np.max(r3_array) - np.min(r3_array))
        red_dc = np.mean(r3_array)
        
        blue_ac = (np.max(b3_array) - np.min(b3_array))
        blue_dc = np.mean(b3_array)
        R = np.mean((red_ac/red_dc)/(blue_ac/blue_dc))
        
        SpO2 = 110-(17*R)
        return SpO2 ,R
    
    '''
    def ac_cal():
        r3_array = []
        b3_array = []
        
        for i in range(len(num_peak_array)-1):
            ac = num_peak_array[i+1]-num_peak_array[i]
            ac_array.append(ac)
        return red_ac_array, blue_ac_array
    
    def dc_cal():
        red_dc_array = []
        blue_dc_array = []
        
        for i in range(len(num_peak_array)-1):
            dc = num_peak_array[i+1]-num_peak_array[i]
            dc_array.append(dc)
        return red_dc_array, blue_dc_array
    '''
    # def ac_dc_cal(self,num_peak_array):
    #     r3_array = []
    #     b3_array = []
    #     for i in range(len(num_peak_array)-1):
    #         r3 = num_peak_array[i+1]-num_peak_array[i]
    #         b3 = num_peak_array[i+1]-num_peak_array[i]
    #         r3_array.append(r3)
    #         b3_array.append(b3)
        
    #     return r3_array, b3_array
    
    def ac_dc_cal(self,num_peak_array,num_vally_array):
        signal_array = []
        for i in range(len(num_peak_array)-1):
            signal = num_peak_array[i]-num_vally_array[i]
            signal_array.append(signal)
        return signal_array
            
    def numPeakFun(self,bp,t1):
        first_peak = 0
        first_peak_bool = False
        bp=bp
        
        #t1= np.arange(100)
        #t1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\time1.txt', delimiter='\n',max_rows =400)
        t1=t1
        #d1= np.arange(99)
        #d1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative1.txt', delimiter='\n',max_rows =297)
        #bp= np.around(x10_7,0)
        
        #d2= np.arange(98)
        #d2= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative2.txt', delimiter='\n',max_rows =294)
        #bp= np.around(x10_7,0)
        #print(len(bp),' ',len(t1),'  ',len(d1),'  ',len(d2))
        
        #print(len(bp))
        #歸一化
        avg = sum(bp)/len(bp)
        bp =self.Z_ScoreNormalization(bp,avg,statistics.stdev(bp))

        #時間
        for i in t1:
            i=i/10
        #print(t1)
        #t1=t1/10
        bp=signal.detrend(bp)
        #print('t1長度:',len(t1))
        #一階差分刪掉第一個時間
        t2=t1
        t2=np.delete(t2,0)
        #print('t2長度:',len(t2))
        #二階差分刪掉1,2值
        t3=t1
        t3=np.delete(t3,0)
        t3=np.delete(t3,0)
        #print('t3長度:',len(t3))
        
        #一階導數
        d1=self.get_derivative1(bp)
        #一階差分
        #d1=[bp[i]-bp[i+1] for i in range(len(bp)-1)]
        #print('d1長度:',len(d1))
        #二階差分
        d2=[bp[i]-bp[i+1]-bp[i+2] for i in range(len(bp)-2)]
        #print('d2長度:',len(d2))
        
        #二階差分抓最高值
        num_peak_3 = signal.find_peaks(d2, distance=None)#distance表極大值點的距離至少大於等於10個水平單位
        #print(len(num_peak_3))
        
        #print(len(num_peak_3[0]))
        if len(num_peak_3[0])>=2:
            #print(num_peak_3[0])
            #print('the number of peaks is ' + str(len(num_peak_3[0])))
            #輸入進陣列放到圖上
            min1=[]
            min1.append(num_peak_3[0][0])
            for i in range(len(num_peak_3)):
                if len(num_peak_3[0])==i:
                    min1.append(num_peak_3[0][i-1])
                    #print(min1)
            
                
            #抓極值 二階差分最高值
            #num_peak_3 = signal.find_peaks(d2, distance=None)
            #print(num_peak_3[0])
            #輸入進陣列放到圖上

            max1_d2=[]
            max1_t3=[]
            
            for i in range(len(num_peak_3[0])):
              if len(num_peak_3[0])==i: 
                max1_d2.append(d2[num_peak_3[0][0]])
                max1_d2.append(d2[num_peak_3[0][0]])
                max1_t3.append(t3[num_peak_3[0][i-1]])
                max1_t3.append(t3[num_peak_3[0][i-1]])
                #抓到一個週期的索引值 64~218
                #print('A波位置',num_peak_3[0][2],'下一個',num_peak_3[0][3])
                
                #抓一階導數一個周期圖上的點
                max1_d1=[]
                max1_t1=[]
                max1_d1.append(d1[num_peak_3[0][0]])
                max1_d1.append(d1[num_peak_3[0][i-1]+1])
                max1_t1.append(t1[num_peak_3[0][0]])
                max1_t1.append(t1[num_peak_3[0][i-1]+1])
            
            '''
            #抓一階差分一個周期
            max1_d1=[]
            max1_t2=[]
            max1_d1.append(d1[num_peak_3[0][2]+1])
            max1_d1.append(d1[num_peak_3[0][4]+1])
            max1_t2.append(t2[num_peak_3[0][2]+1])
            max1_t2.append(t2[num_peak_3[0][4]+1])
            '''
            
            #抓一階倒數的波
            
            if len(num_peak_3[0])==2: 
                Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                #Tbp=smoothTriangle(Tbp)
                TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                #print('Tbp長度:',len(Tbp))
                #print('TT1長度:',len(TT1))
            else:
                for j in range(len(num_peak_3[0])):
                  if len(num_peak_3)==j: 
                    Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                    #Tbp=smoothTriangle(Tbp)
                    TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                    #print('Tbp長度:',len(Tbp))
                    #print('TT1長度:',len(TT1))
            
            
            #from scipy.interpolate import interp1d #注意是數字的1
            #f1= interp1d(TT1,Tbp)               #產生線性插值函數
            #print('max',max(TT1))
            #print('min',min(TT1))
            #x = np.linspace(1.7,2.67,100)             #將間隔細分為50個區段
            #y = f1(x)                              #利用線性插值函數產生50個插值
            #print(y)
            #plt.plot(TT1,Tbp,'b^',x, y, "ro", label='linear interplot')
            
            
            #64~218共155個值作為一個週期
            #print('一階差分',len(TT1),len(Tbp))
            
            
            #抓波峰
            num_peak_4 = signal.find_peaks(Tbp, distance=None)
            first_peak_bool = True
            try:
                first_peak = TT1[num_peak_4[0][0]]
                #print("first peak: " + str(first_peak))
            except:
                first_peak_bool = False    
            
            if len(num_peak_4[0])==1:
               first_peak_bool = False
               
        return first_peak, first_peak_bool
    
    ##抓波谷
    def numVallyFun(self,Signal):
        first_vally = 0
        first_vally_bool = False
        arrSignal = np.array(Signal)

        num_vally_1 = signal.find_peaks(-arrSignal, distance=None)
        first_vally_bool = True
        try:
            first_vally = num_vally_1[0][0]
            #print("first vally: " + str(first_vally))
        except:
            first_vally_bool = False    
        
        if len(num_vally_1[0])==1:
            first_vally_bool = False

        return first_vally, first_vally_bool