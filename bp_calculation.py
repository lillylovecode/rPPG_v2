import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import statistics


class BP:
    def __init__(self):
        self.SBP_DBP = []
        self.SampEn_timedata = []
        self.SampEn_time = 0
        self.allPeakTimeData = []

    def closest(self, mylist, number):
        answer = [abs(number - i) for i in mylist]
        return answer.index(min(answer))

    def Z_ScoreNormalization(self, x, mu, sigma):
        return (x - mu) / sigma

    def get_derivative1(self, x, maxiter=1):
        h = 0.0001
        F = lambda x: 0.5 * x ** 2 * (4 - x)

        def df(x, f=F):
            return (f(x + h) - f(x)) / h

        for _ in range(maxiter):
            x = x - F(x) / df(x)
        return x

    def wave_guess(self, arr, t3):
        wn = int(len(arr) / 4)
        wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x: x[1])
        wave_crest_mean = pd.DataFrame(wave_crest).mean()

        wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x: x[1])
        wave_base_mean = pd.DataFrame(wave_base).mean()

        wave_period = abs(int(wave_crest_mean[0] - wave_base_mean[0]))
        print("wave_period_day:", wave_period)
        print("wave_crest_mean:", round(wave_crest_mean[1], 2))
        print("wave_base_mean:", round(wave_base_mean[1], 2))

        wave_crest_x = [i for i, _ in wave_crest]
        wave_crest_y = [j for _, j in wave_crest]

        wave_base_x = [i for i, _ in wave_base]
        wave_base_y = [j for _, j in wave_base]

        plt.figure(figsize=(12, 3))
        plt.plot(arr)
        plt.plot(wave_crest_x, wave_crest_y, 'ro')
        plt.grid()
        plt.show()

    def smoothTriangle(self, data, degree=1):
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
        smoothed = []

        for i in range(degree, len(data) - degree * 2):
            point = data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point) / np.sum(triangle))

        smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed

    def run(self, bp, t1):
        bp = self.Z_ScoreNormalization(bp, sum(bp) / len(bp), statistics.stdev(bp))
        t1 = [i / 10 for i in t1]

        bp = signal.detrend(bp)
        t2 = np.delete(t1, 0)
        t3 = np.delete(np.delete(t1, 0), 0)

        d1 = self.get_derivative1(bp)
        d2 = [bp[i] - bp[i + 1] - bp[i + 2] for i in range(len(bp) - 2)]

        num_peak_3 = signal.find_peaks(d2)[0]

        if len(num_peak_3) >= 2:
            min1 = [num_peak_3[0]]
            if len(num_peak_3) == len(min1):
                min1.append(num_peak_3[-1])

            max1_d2 = [d2[i] for i in num_peak_3]
            max1_t3 = [t3[i] for i in num_peak_3]

            max1_d1 = [d1[num_peak_3[0]], d1[num_peak_3[-1] + 1]]
            max1_t1 = [t1[num_peak_3[0]], t1[num_peak_3[-1] + 1]]

            if len(num_peak_3) == 2:
                Tbp = [d1[i] for i in range(num_peak_3[0], num_peak_3[1])]
                TT1 = [t2[i] for i in range(num_peak_3[0], num_peak_3[1])]
            else:
                for j in range(len(num_peak_3)):
                    if len(num_peak_3) == j:
                        Tbp = [d1[i] for i in range(num_peak_3[0], num_peak_3[j])]
                        TT1 = [t2[i] for i in range(num_peak_3[0], num_peak_3[j])]

            num_peak_4 = signal.find_peaks(Tbp)[0]

            if len(num_peak_4) == 1:
                return 0, 0

            max1_Tbp = [Tbp[num_peak_4[0]], Tbp[num_peak_4[1]]]
            max1_TT1 = [TT1[num_peak_4[0]], TT1[num_peak_4[1]]]

            num_zero = [self.closest(Tbp[0:35], 0)]
            zero_Tbp = [Tbp[i] for i in num_zero]
            zero_TT1 = [TT1[i] for i in num_zero]

            minTbp_num = min(Tbp[50:100])
            b = [i + 50 for i in range(len(Tbp[50:100])) if Tbp[i + 50] == minTbp_num][0]

            min_Tbp = [Tbp[b]]
            min_TT1 = [TT1[b]]

            feature_d1 = [Tbp[0], Tbp[num_zero[0]], Tbp[num_peak_4[0]], Tbp[b], Tbp[num_peak_4[1]], Tbp[-1]]
            feature_Time = [TT1[0], TT1[num_zero[0]], TT1[num_peak_4[0]], TT1[b], TT1[num_peak_4[1]], TT1[-1]]

            time1 = TT1[-1] - TT1[0]
            time2 = 60 / time1
            SBP = -141.3 * ((TT1[num_peak_4[1]] - TT1[0]) / time1) + 0.68 * time1 * time2 + 145.6
            DBP = -93.2 * ((TT1[num_peak_4[1]] - TT1[0]) / time1) + 0.15 * time1 * time2 + 120.6

            self.SampEn_timedata.append(TT1[num_peak_4[1]] - self.SampEn_time)
            self.SampEn_time = TT1[num_peak_4[1]]
            return SBP, DBP
        else:
            return 0.0, 0.0
