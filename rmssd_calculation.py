import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from scipy import signal


class RMSSD_calculation(object):
    def __init__(self):
        self.allPeakTimeData = []
        self.ppi_array1 = []
        self.ppi_minus = []

    def run(self, bp, t1):
        rmssd_bool = False
        rmssd = 0
        allPeakTimeDataLength = 100  # 這個值決定要用多少長度的allPeakTimeData[]計算RMSSD，可以調整
        first_peak, first_peak_bool = self.numPeakFun(bp, t1)
        if first_peak_bool:
            self.allPeakTimeData.append(first_peak)
            if len(self.allPeakTimeData) >= allPeakTimeDataLength:
                currentPeakTimeData = self.allPeakTimeData[-allPeakTimeDataLength:]
                ppi_array = self.ppi_cal(currentPeakTimeData)
                ppi_array1 = self.ppi_cal(currentPeakTimeData)
                rmssd = self.rmssd_fun(ppi_array, ppi_array1)
                self.allPeakTimeData = currentPeakTimeData
                rmssd_bool = True
        return rmssd, rmssd_bool

    def closest(self, mylist, Number):
        answer = [abs(Number - i) for i in mylist]
        return answer.index(min(answer))

    def Z_ScoreNormalization(self, x, mu, sigma):
        return (x - mu) / sigma

    def get_derivative1(self, x, maxiter=1):
        h = 0.0001
        F = lambda x: 0.5 * x**2 * (4 - x)

        def df(x, f=F):
            return (f(x + h) - f(x)) / h

        for i in range(maxiter):
            x = x - F(x) / df(x)
        return x

    def smoothTriangle(self, data, degree=1):
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
        smoothed = []

        for i in range(degree, len(data) - degree * 2):
            point = data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point) / np.sum(triangle))

        smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed

    def rmssd_fun(self, ppi_array, ppi_array1):
        ppi_array1 = np.array(self.ppi_minus)
        rmssd = np.sqrt(sum((ppi_array[i + 1] - ppi_array[i])**2 for i in range(len(ppi_array) - 1)) / len(ppi_array))
        return rmssd

    def ppi_cal(self, num_peak_array):
        ppi_array = []
        for i in range(len(num_peak_array) - 1):
            ppi = num_peak_array[i + 1] - num_peak_array[i]
            ppi_array.append(ppi)
        return ppi_array

    def numPeakFun(self, bp, t1):
        first_peak = 0
        first_peak_bool = False

        avg = sum(bp) / len(bp)
        bp = self.Z_ScoreNormalization(bp, avg, statistics.stdev(bp))

        t1 = [i / 10 for i in t1]

        bp = signal.detrend(bp)

        t2 = np.delete(t1, 0)
        t3 = np.delete(t1, [0, 1])

        d1 = self.get_derivative1(bp)
        d2 = [bp[i] - bp[i + 1] - bp[i + 2] for i in range(len(bp) - 2)]

        num_peak_3 = signal.find_peaks(d2, distance=None)

        if len(num_peak_3[0]) >= 2:
            min1 = [num_peak_3[0][0]] + [num_peak_3[0][i - 1] for i in range(1, len(num_peak_3[0]))]

            max1_d2 = [d2[num_peak_3[0][0]], d2[num_peak_3[0][-1]]]
            max1_t3 = [t3[num_peak_3[0][0]], t3[num_peak_3[0][-1]]]

            max1_d1 = [d1[num_peak_3[0][0]], d1[num_peak_3[0][-1] + 1]]
            max1_t1 = [t1[num_peak_3[0][0]], t1[num_peak_3[0][-1] + 1]]

            if len(num_peak_3[0]) == 2:
                Tbp = [d1[i] for i in range(num_peak_3[0][0], num_peak_3[0][1])]
                TT1 = [t2[i] for i in range(num_peak_3[0][0], num_peak_3[0][1])]
            else:
                Tbp = [d1[i] for i in range(num_peak_3[0][0], num_peak_3[0][-1])]
                TT1 = [t2[i] for i in range(num_peak_3[0][0], num_peak_3[0][-1])]

            num_peak_4 = signal.find_peaks(Tbp, distance=None)
            first_peak_bool = True
            try:
                first_peak = TT1[num_peak_4[0][0]]
            except:
                first_peak_bool = False

            if len(num_peak_4[0]) == 1:
                first_peak_bool = False

        return first_peak, first_peak_bool
