import numpy as np
import time
import statistics
import random
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import distance as dist
from sklearn.decomposition import FastICA
from face_detection import FaceDetection
from bp_calculation import BP
from sdnn_calculation import SDNN_calculation
from rmssd_calculation import RMSSD_calculation
from SpO2 import SpO2_calculation
from saving import write_to_file, saving

class Process:
    def __init__(self):
        self.init_variables(self)
        self.fd = FaceDetection()
        self.BP = BP()
        self.sdnn_cal = SDNN_calculation()
        self.rmssd_cal = RMSSD_calculation()
        self.SpO2_cal = SpO2_calculation()

    @staticmethod
    def init_variables(self):

        ## 人臉偵測
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.fps = 0
        self.ROIidx=0

        ## 訊號處理
        self.samples = []
        self.samplesPPG = []
        self.samplesPPG2 = []
        self.processed_G = []
        self.processed_R = []
        self.processed_B = []
        self.bpms = []
        self.bpm = 0
        self.bp=[]  # 綠色通道(平滑化後)
        self.bpR=[] # 紅色通道(平滑化後)
        self.bpB=[] # 藍色通道(平滑化後)
        self.SBP_DBP = 0
        self.SBP_DBP2 = 0
        self.fft = []

        ## 特徵值
        self.lfhf_data = []
        self.hf_data = []
        self.lf_data = []
        self.SpO2_data = []
        self.R_data = []

        ## 時間資料
        self.count = 0
        self.start_time = 0
        self.start_time_status = 0
        self.file = False
        self.buffer_size = 100
        self.buffer_size2 = 1200
        self.time_buffer = []
        self.time_buffer_red=[]
        self.time_buffer2_red=[]
        self.time_buffer_blue=[]
        self.time_buffer2_blue=[]
        self.data_buffer = []
        self.data_buffer2 = []
        self.data_buffer_red = []
        self.data_buffer2_red = []
        self.data_buffer_blue = []
        self.data_buffer2_blue = []
        self.processed_buffers = []
        self.t0 = time.time()
        self.progress = 0

        ## 心理特徵
        self.sdnn=0
        self.rmssd=0
        
        ## LFHF
        self.hf=0
        self.lf=0
        self.hf_area=0
        self.lf_area=0
        self.lfhf=0

        ## 眨眼
        self.blinks_counter=0

    def run(self):
        self.initialize_start_time()
        frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye = self.fd.face_detect(self.frame_in,self.ROIidx)
        self.frame_out = frame
        self.frame_ROI = face_frame
        self.process_frame(ROI1, ROI2)
        #self.update_eye_ratio(leftEye, rightEye)

    def initialize_start_time(self):
        if self.start_time_status == 0:
            self.start_time = time.time()
            self.start_time_status = 1

    def process_frame(self, ROI1, ROI2):
        g1, r1, b1, _ = self.extract_color(ROI1)
        g2, r2, b2, _ = self.extract_color(ROI2)
        g, r, b = (g1 + g2) / 2, (r1 + r2) / 2, (b1 + b2) / 2

        LenOfGreenBuffer = len(self.data_buffer)
        LenOfGreenBuffer2 = len(self.data_buffer2)     
        
        LenOfRedBuffer = len(self.data_buffer_red)
        LenOfRedBuffer2 = len(self.data_buffer2_red)
        
        LenOfBlueBuffer = len(self.data_buffer_blue)
        LenOfBlueBuffer2 = len(self.data_buffer2_blue)
        
        g = (g1+g2)/2 
        r = (r1+r2)/2 
        b = (b1+b2)/2 
        
        g_f=g
        r_f=r
        b_f=b
        
        self.check = False

        #green channel-----------------------------------------------------------
        
        if(abs(g-np.mean(self.data_buffer))>10 and LenOfGreenBuffer>99):
            g = self.data_buffer[-1]
            
        if(abs(g_f-np.mean(self.data_buffer2))>10 and LenOfGreenBuffer2>499):
            g_f = self.data_buffer2[-1]
            
        if g!=0 :
            self.times.append(time.time() - self.t0)
            self.data_buffer.append(g)
           
        #red channel--------------------------------------------------------------
           
        if(abs(r-np.mean(self.data_buffer_red))>10 and LenOfRedBuffer>99):
            r = self.data_buffer_red[-1]
            
        if(abs(r_f-np.mean(self.data_buffer2_red))>10 and LenOfRedBuffer2>499):
            r_f = self.data_buffer2_red[-1]            
            
        if r!=0 :
            self.times_red.append(time.time() - self.t0)
            self.data_buffer_red.append(r)    
            
        #blue channel----------------------------------------------------------
        
        if(abs(b-np.mean(self.data_buffer_blue))>10 and LenOfBlueBuffer>99):
            b = self.data_buffer_blue[-1]
            
        if(abs(b_f-np.mean(self.data_buffer2_blue))>10 and LenOfBlueBuffer2>499):
            b_f = self.data_buffer2_blue[-1]
            
        if b!=0 :
            self.times_blue.append(time.time() - self.t0)
            self.data_buffer_blue.append(b)  


        #green channel----------------------------------------------------------

        if len(self.time_buffer)>0:
                self.time_buffer.append(abs(round(time.time()-self.time_buffer[0],4)))
        else:
            self.time_buffer.append(time.time()) 
            
        if len(self.time_buffer)>301:
            self.time_buffer.pop(1)
        if LenOfGreenBuffer > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                self.bpms = self.bpms[-self.buffer_size//2:]
                LenOfGreenBuffer = self.buffer_size
                
        processed = np.array(self.data_buffer)
        
        #red channel-------------------------------------------------------------
    
        if len(self.time_buffer_red)>0:
            self.time_buffer_red.append(abs(round(time.time()-self.time_buffer_red[0],4)))
        else:
            self.time_buffer_red.append(time.time()) 
            
        if len(self.time_buffer_red)>301:
            self.time_buffer_red.pop(1)
        if LenOfRedBuffer > self.buffer_size_red:
                self.data_buffer_red = self.data_buffer_red[-self.buffer_size_red:]
                self.times_red = self.times_red[-self.buffer_size_red:]
                self.bpms = self.bpms[-self.buffer_size_red//2:]
                LenOfRedBuffer = self.buffer_size_red
                
        processed_red = np.array(self.data_buffer_red)
            
        #blue channel--------------------------------------------------------------

        if len(self.time_buffer_blue)>0:
            self.time_buffer_blue.append(abs(round(time.time()-self.time_buffer_blue[0],4)))
        else:
            self.time_buffer_blue.append(time.time()) 
            
        if len(self.time_buffer_blue)>301:
            self.time_buffer_blue.pop(1)
        if LenOfBlueBuffer > self.buffer_size_blue:
                self.data_buffer_blue = self.data_buffer_blue[-self.buffer_size_blue:]
                self.times_blue = self.times_blue[-self.buffer_size_blue:]
                self.bpms = self.bpms[-self.buffer_size_blue//2:]
                LenOfBlueBuffer = self.buffer_size_blue
                
        processed_blue = np.array(self.data_buffer_blue)


        self.update_buffers(g, r, b)
        self.calculate(g, r, b)

    def extract_color(self, frame):
        b = np.mean(frame[:, :, 0])
        g = np.mean(frame[:, :, 1])
        r = np.mean(frame[:, :, 2])
        return g, r, b, (r + g + b) / 3
    
    def ICA(self,norm):
        X=(norm,[1*random.random() for i in range(len(norm))])
        ICA = FastICA(n_components=1,max_iter=1000,tol=1e-6)
        X_transformed = ICA.fit_transform(X)
        A_ =  ICA.mixing_.T
        return A_

    def update_buffers(self, g, r, b):
        self.update_buffer('green', g)
        self.update_buffer('red', r)
        self.update_buffer('blue', b)

    def update_buffer(self, color, value):
        if color == 'green':
            buffer = self.data_buffer
            if abs(value - np.mean(buffer)) > 10 and len(buffer) > 99:
                value = buffer[-1]
            self.time_buffer.append(time.time() - self.t0)
            buffer.append(value)
            if len(buffer) > self.buffer_size:
                buffer.pop(0)
                self.time_buffer.pop(0)
        elif color == 'red':
            buffer = self.data_buffer_red
            if abs(value - np.mean(buffer)) > 10 and len(buffer) > 99:
                value = buffer[-1]
            self.time_buffer_red.append(time.time() - self.t0)
            buffer.append(value)
            if len(buffer) > self.buffer_size:
                buffer.pop(0)
                self.time_buffer_red.pop(0)
        else:
            buffer = self.data_buffer_blue
            if abs(value - np.mean(buffer)) > 10 and len(buffer) > 99:
                value = buffer[-1]
            self.time_buffer_blue.append(time.time() - self.t0)
            buffer.append(value)
            if len(buffer) > self.buffer_size:
                buffer.pop(0)
                self.time_buffer_blue.pop(0)

    def calculate(self, g, r, b):
        self.preprocessing(g, 'green')
        self.preprocessing(r, 'red')
        self.preprocessing(b, 'blue')
        self.calculate_spO2()
        self.calculate_lfhf()
        self.save_data()

    def preprocessing(self, value, color):
        
        # 初步訊號處理(去趨勢、butterworth濾波)
        processed = self.process_buffer(self.data_buffer if color == 'green' else (self.data_buffer_red if color == 'red' else self.data_buffer_blue))













        # 檢查buffer是否已滿
        if len(self.data_buffer) == self.buffer_size:
            L = len(self.data_buffer)
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)

            #ICA
            #A_=self.ICA(processed)
            #processed =A_[0]

            self.data_buffer2.append(processed)
            self.bpms.append(self.get_bpm(processed))

        if len(self.data_buffer_red) == self.buffer_size:
            P = len(self.data_buffer_red)
            self.data_buffer2_red.append(processed)
            

        if len(self.data_buffer_blue) == self.buffer_size:
            K = len(self.data_buffer_blue)
            self.data_buffer2_blue.append(processed)
            

    def process_buffer(self, buffer):
        buffer = np.array(buffer)
        buffer = signal.detrend(buffer)
        buffer = self.butterworth_bandpass_filter(buffer, 0.8, 3, len(buffer), order=4)
        return buffer

    def get_bpm(self, buffer):
        raw = np.fft.rfft(buffer)
        freqs = 60. * np.fft.fftfreq(len(buffer), d=1.0)
        fft = np.abs(raw)**2
        idx = np.where((freqs > 60) & (freqs < 100))
        pruned = fft[idx]
        pfreq = freqs[idx]
        return pfreq[np.argmax(pruned)]

    def calculate_spO2(self):
        if len(self.time_buffer) > 300:
            spO2_temp, spO2_bool, R = self.SpO2_cal.run(self.processed_buffers['red'], self.processed_buffers['blue'], self.time_buffer)
            if spO2_bool and self.count % 50 == 0:
                self.SpO2 = spO2_temp
                self.SpO2_data.append(self.SpO2)
                self.R = R
                self.R_data.append(self.R)

    def calculate_lfhf(self):
        if self.count % 50 == 0:
            raw = np.fft.rfft(self.processed_buffers)
            freqs = 60. * np.fft.fftfreq(len(self.processed_buffers), d=1.0)
            fft = np.abs(raw)**2
            self.lf, self.hf = self.calculate_area(freqs, fft)
            self.lfhf = self.lf / self.hf
            self.lfhf_data.append(self.lfhf)

    def calculate_area(self, freqs, fft):
        lf_idx = np.where((freqs >= 0.04) & (freqs <= 0.15))
        hf_idx = np.where((freqs > 0.15) & (freqs < 0.4))
        lf_area = np.trapz(fft[lf_idx], freqs[lf_idx])
        hf_area = np.trapz(fft[hf_idx], freqs[hf_idx])
        return lf_area / (lf_area + hf_area), hf_area / (lf_area + hf_area)

    def save_data(self):
        if self.count % 100 == 0:
            filename = "result_" + time.strftime("%Y%m%d%H%M")
            saving(filename, self.collect_results())

    def collect_results(self):
        results = {
            'bpm': self.bpms,
            'SpO2': self.SpO2_data,
            'R': self.R_data,
            'lfhf': self.lfhf_data,
        }
        return results

    ##巴特濾波器
    def butterworth_bandpass(self, lowcut, highcut, fs, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band',analog='True')
        return b, a
        #order 是濾波器的階數，階數越大，濾波效果越好，但是計算量也會跟著變大。
        #所產生的濾波器參數 a 和 b 的長度，等於 order+1。
        #Wn 是正規化的截止頻率，介於 0 和 1 之間，當取樣頻率是 fs 時，所能處理的
        #最高頻率是 fs/2，所以如果實際的截止頻率是 f = 1000，那麼 Wn = f/(fs/2)。
        #function 是一個字串，function = 'low' 代表是低通濾波器，function = 'high' 代表是高通濾波。
        #fs=12,wn=f/(fs/2),如果截止頻率大於6,就高於正規化的截止頻率

    def butterworth_bandpass_filter(self, data, lowcut, highcut, fs, order=1):
        b, a = self.butterworth_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def update_eye_ratio(self, leftEye, rightEye):
        leftEAR = self.eye_ratio(leftEye)
        rightEAR = self.eye_ratio(rightEye)
        self.blinks_counter, self.blinks = self.calculate_blinks(leftEAR, rightEAR, self.blinks_counter, self.blinks)

    def eye_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0

    def calculate_blinks(self, leftEAR, rightEAR, blinks_counter, blinks):
        eye_ar_thresh = 0.3
        eye_ar_consec_frames = 3
        if leftEAR != 0 and rightEAR != 0:
            ear = (leftEAR + rightEAR) / 2.0
            if ear < eye_ar_thresh:
                blinks_counter += 1
            else:
                if blinks_counter >= eye_ar_consec_frames:
                    blinks += 1
                blinks_counter = 0
        return blinks_counter, blinks

    def reset(self):
        self.init_variables(self)
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
