import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from scipy import signal


class SDNN_calculation(object):
    def __init__(self):
        self.allPeakTimeData=[]
        
    def run(self,bp,t1,bp1,t2):
        sdnn_bool = False
        rmssd_bool = False
        sdnn = 0
        rmssd = 0
        
        allPeakTimeDataLength = 100 #這個值決定要用多少長度的allPeakTimeData[]計算SDNN，可以調整
        first_peak, first_peak_bool = self.numPeakFun(bp,t1)
        if(first_peak_bool==True):
            self.allPeakTimeData.append(first_peak)                
            if(len(self.allPeakTimeData)>=allPeakTimeDataLength):
                currentPeakTimeData = self.allPeakTimeData[-allPeakTimeDataLength:]
                
                ppi_array = self.ppi_cal(currentPeakTimeData)
                sdnn = self.sdnn_fun(ppi_array)
                rmssd = self.rmssd_fun(ppi_array)
                self.allPeakTimeData = currentPeakTimeData
                sdnn_bool = True
                rmssd_bool = True
                
        return sdnn, sdnn_bool, rmssd, rmssd_bool
    
    def closest(self,mylist, Number):
        answer = []
        for i in mylist:    
            answer.append(abs(Number-i))
        return answer.index(min(answer))

            #歸一化
    def Z_ScoreNormalization(self,x,mu,sigma):
        x = (x - mu) / sigma;
        return x;
             #抓一階倒數
    def get_derivative1(self,x,maxiter=1):   #//默認使用牛頓法迭代10次
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
    def smoothTriangle(data, degree=1):
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
                 
        
    def sdnn_fun(self,ppi_array):
        mean = np.mean(ppi_array)
        sdnn = np.sqrt((sum((ppi_array - mean)**2)) / (len(ppi_array) - 1))
        return sdnn   
    '''
    def rmssd_fun(self,ppi_array_minus,ppi_array):
        rmssd = np.sqrt((sum((ppi_array_minus)**2)) / (len(ppi_array)-1))        
        return rmssd
    '''
    def rmssd_fun(self,ppi_array):
        ppi_array_diff = np.diff(ppi_array)
        return np.std(ppi_array_diff)
        
    def ppi_cal(self,num_peak_array):        
        ppi_array=[]        
        for i in range(len(num_peak_array)-1):
            ppi = num_peak_array[i+1]-num_peak_array[i]
            ppi_array.append(ppi)
        return ppi_array
    '''
    def ppi_cal_minus(self,ppi_array):
        ppi_array = []
        ppi_array_minus = []
        for i in range(len(ppi_array)-1):
            ppi_minus = ppi_array[i+1] - ppi_array[i]
            ppi_array_minus.append(ppi_minus)
        ppi_array_minus = np.array(ppi_array_minus)
        return ppi_array_minus
    '''
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
            

              
 
            
            
            
            
            
            