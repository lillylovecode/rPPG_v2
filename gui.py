import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import time
import math
import sys
from components import *
from process import Process
from SpO2 import SpO2_calculation
from webcam import Webcam
from interface import waitKey
import saving

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI, self).__init__()
        self.initUI()
        self.webcam = Webcam()
        self.input = self.webcam
        self.dirname = ""
        #self.btnOpen.setEnabled(False)
        self.process = Process()
        self.SpO2 = SpO2_calculation()
        self.status = False
        self.frame = np.zeros((10, 10, 3), np.uint8)
        self.bpm = 0
        self.close_reason = False

    def initUI(self):
        # 設定字體
        font = QFont()
        font.setPointSize(12)          
        
        font2 = QFont()
        font2.setPointSize(14)         #中間輸入文字字體大小
        
        font3 = QFont()
        font3.setPointSize(15)         #所有圖形標題字體大小
        
        font4 = QFont()
        font4.setPointSize(16)         #小視訊框下面的字體大小

        font5 = QFont()
        font5.setPointSize(20)         #回饋區文字字體大小

        # 建立按鈕
        self.btnStart = self.create_button("Start", 1350, 550, 540, 70, font4, self.run)
        #self.btnOpen = self.create_button("Open", 10, 10, 1, 1, font4, self.openFileDialog)

        # 创建标签页
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setGeometry(10, 10, 1000, 900)
        self.tabWidget.setFont(font)

        # 创建各个标签页
        self.create_tab("訊號動圖", font)
        self.create_tab("預測數據", font, self.set_up_statistic_data)
        self.create_tab("前處理(G)", font)
        self.create_tab("前處理(R)", font)
        self.create_tab("前處理(B)", font)
        self.create_tab("回饋區", font, self.setup_feedback_tab)

        # 基本資訊输入框
        self.setup_basic_info()

        # 显示区域
        self.setup_display_area()

        # 定时器
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(300)

        # 狀態欄
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        # 信号
        self.c = Communicate()
        self.c.closeApp.connect(self.close)

        # 窗口设置
        self.setGeometry(10, 100, 1900, 950)
        self.setStyleSheet("background-color:#D0D0D0")
        self.setWindowTitle("可攜式非接觸即時壓力偵測系統")
        self.show()

        # 提示信息
        #show_warning_message()

    def create_button(self, text, x, y, width, height, font, callback):
        button = QPushButton(text, self)
        button.move(x, y)
        button.setFixedWidth(width)
        button.setFixedHeight(height)
        button.setFont(font)
        button.clicked.connect(callback)
        return button

    def create_tab(self, title, font, setup_func=None):
        tab = QWidget()
        self.tabWidget.addTab(tab, title)
        tab.setFont(font)
        tab.setFixedSize(1000, 900)
        tab.setStyleSheet("background-color:#D0D0D0")
        if setup_func:
            setup_func(tab, font)
        return tab

    def setup_feedback_tab(self, tab, font):
        font2 = QFont()
        font2.setPointSize(14)
        font4 = QFont()
        font4.setPointSize(16)
        font5 = QFont()
        font5.setPointSize(20)

        # 设置回馈区的各个输入框和标签
        create_label(tab, "回饋區", 10, 10, 150, 60, font5)
        self.input_HR = create_input_field(tab, "HR", 10, 90, "請輸入HR", "input_HR", font2, font4)
        self.input_SBP = create_input_field(tab, "SBP", 10, 190, "請輸入SBP", "input_SBP", font2, font4)
        self.input_DBP = create_input_field(tab, "DBP", 10, 290, "請輸入DBP", "input_DBP", font2, font4)
        self.input_RR = create_input_field(tab, "RR", 10, 390, "請輸入RR", "input_RR", font2, font4)
        create_combobox(tab, "譫妄", 400, 90, ["有無明顯譫妄", "明顯譫妄", "無明顯譫妄"], font2)
        create_input_field(tab, "Hb", 400, 190, "g/dL", "input_Hb", font2, font4)
        create_input_field(tab, "膽色素", 400, 290, "mg/dL", "input_Bilirubin", font2, font4)
        create_combobox(tab, "抽血", 400, 390, ["無", "開始抽血", "結束抽血"], font2)
        create_input_field(tab, "SPO2", 10, 550, "請輸入SPO2", "input_SPO2", font2, font4)
        create_input_field(tab, "G/S", 10, 610, "請輸入血糖", "input_Glucose", font2, font4)

        self.btnSave = QPushButton("Save", tab)
        self.btnSave.move(10, 700)
        self.btnSave.setFixedWidth(540)
        self.btnSave.setFixedHeight(70)
        self.btnSave.setFont(font2)
        self.btnSave.clicked.connect(self.saveData)

        self.btnPsychological = QPushButton("心理評估", tab)
        self.btnPsychological.move(530, 600)
        self.btnPsychological.setFixedWidth(200)
        self.btnPsychological.setFixedHeight(60)
        self.btnPsychological.setFont(font2)
        self.btnPsychological.clicked.connect(self.openPsyDialog)
    
    def setup_basic_info(self):
        first_row = 700
        self.input_num = create_basic_info_input(self, "病歷號/編號", 1350, first_row)
        self.input_name = create_basic_info_input(self, "請輸入姓名", 1530, first_row)
        self.cbb_gender = create_basic_info_combobox(self, ["性別", "男", "女"], 1710, first_row)
        self.cbbCamera = create_basic_info_combobox(self, ["內建鏡頭", "外接鏡頭"], 1100, 245, self.selectCamera)
        self.cbbROI = create_basic_info_combobox(self, ["全臉", "額頭", "下巴", "臉頰"], 1100, 345, self.selectROI)

        self.input_count = QLineEdit(self)
        self.input_count.move(630, 1100)
        self.input_count.setFixedWidth(1)
        self.input_count.setFixedHeight(1)
        self.input_count.setFont(QFont())
        self.input_count.setText("1000")
        self.input_count.setPlaceholderText("請勿隨意修改!")
        self.input_count.setObjectName("input_count")
        self.input_count.setEchoMode(QLineEdit.PasswordEchoOnEdit)

    def set_up_statistic_data(self, tab, font):
        geometry_statistics_x = 15
        geometry_statistics_x2 = 500

        font4 = QFont()
        font4.setPointSize(16)

        # 使用 create_label 函數創建標籤
        self.lblHR2 = create_label(tab, "Heart rate: ", geometry_statistics_x, 5, 300, 70, font4)
        self.lblHF = create_label(tab, "HF(nu): ", geometry_statistics_x, 100, 300, 70, font4)
        self.lblLF = create_label(tab, "LF(nu): ", geometry_statistics_x, 200, 300, 70, font4)
        self.lblLFHF = create_label(tab, "LF/HF: ", geometry_statistics_x, 300, 300, 70, font4)
        self.lblBP = create_label(tab, "Blood Pressure: ", geometry_statistics_x, 400, 300, 70, font4)
        self.lblsdnn = create_label(tab, "SDNN: ", geometry_statistics_x, 500, 300, 70, font4)
        self.lblrmssd = create_label(tab, "RMSSD: ", geometry_statistics_x, 600, 300, 70, font4)
        self.lblSPO2 = create_label(tab, "SpO2: ", geometry_statistics_x, 700, 300, 70, font4)
        self.lblR = create_label(tab, "R: ", geometry_statistics_x, 800, 300, 70, font4)
        self.lblRR2 = create_label(tab, "Respiratory rate: ", geometry_statistics_x2, 5, 300, 70, font4)
        self.lbltotaltime = create_label(self, "Total Time: ", 1350, 450, 300, 80, font4)

    def setup_display_area(self):

        #小視訊框
        self.lblDisplay = QLabel(self)
        self.lblDisplay.setGeometry(1350, 10, 540, 420)
        self.lblDisplay.setStyleSheet("background-color: #000000")

        #ROI圈選視窗
        self.lblROI = QLabel(self)
        self.lblROI.setGeometry(1100, 10, 200, 200)
        self.lblROI.setStyleSheet("background-color: #000000")

        # 信號動圖
        self.signal_Plt = create_plot(self.tabWidget.widget(0), 15, 31, 470, 192, '#FFFFF0', "rPPG_G")
        self.signal2_Plt = create_plot(self.tabWidget.widget(0), 15, 254, 470, 192, '#FFFFF0', "rPPG_R")
        self.signal3_Plt = create_plot(self.tabWidget.widget(0), 15, 477, 470, 192, '#FFFFF0', "rPPG_B")

        self.fft_Plt_G = create_plot(self.tabWidget.widget(0), 550, 31, 470, 192, '#FFFFF0', "FFT_G")
        self.fft_Plt_R = create_plot(self.tabWidget.widget(0), 550, 254, 470, 192, '#FFFFF0', "FFT_R")
        self.fft_Plt_B = create_plot(self.tabWidget.widget(0), 550, 477, 470, 192, '#FFFFF0', "FFT_B")

        # 前處理(G)
        self.signal_Plt1 = create_plot(self.tabWidget.widget(2), 15, 31, 470, 192, '#FFFFF0', "原始訊號(橫軸單位：時間)")
        self.signal_process_Plt1 = create_plot(self.tabWidget.widget(2), 15, 254, 470, 192, '#FFFFF0', "基線校正(橫軸單位：時間)")
        self.signal_smooth_Plt1 = create_plot(self.tabWidget.widget(2), 15, 477, 470, 192, '#FFFFF0', "歸一化(橫軸單位：時間)")

        # 前處理(R)
        self.signal_Plt2 = create_plot(self.tabWidget.widget(3), 15, 31, 470, 192, '#FFFFF0', "原始訊號(橫軸單位：時間)")
        self.signal_process_Plt2 = create_plot(self.tabWidget.widget(3), 15, 254, 470, 192, '#FFFFF0', "基線校正(橫軸單位：時間)")
        self.signal_smooth_Plt2 = create_plot(self.tabWidget.widget(3), 15, 477, 470, 192, '#FFFFF0', "歸一化(橫軸單位：時間)")

        # 前處理(B)
        self.signal_Plt3 = create_plot(self.tabWidget.widget(4), 15, 31, 470, 192, '#FFFFF0', "原始訊號(橫軸單位：時間)")
        self.signal_process_Plt3 = create_plot(self.tabWidget.widget(4), 15, 254, 470, 192, '#FFFFF0', "基線校正(橫軸單位：時間)")
        self.signal_smooth_Plt3 = create_plot(self.tabWidget.widget(4), 15, 477, 470, 192, '#FFFFF0', "歸一化(橫軸單位：時間)")

    def update(self):
        # 更新图像和数据
        self.update_plots()

    def update_plots(self):
        # 更新各个图像和数据
        self.signal_Plt.clear()
        self.signal_Plt.plot(self.process.bp, pen=pg.mkPen('#55C355', width=2))

        self.signal2_Plt.clear()
        self.signal2_Plt.plot(self.process.bpR, pen=pg.mkPen('#FF0000', width=2))

        self.signal3_Plt.clear()
        self.signal3_Plt.plot(self.process.bpB, pen=pg.mkPen('#0000FF', width=2))

        self.fft_Plt_B.clear()
        self.fft_Plt_B.plot(self.process.fft, pen=pg.mkPen('#000000', width=2))

        self.signal_Plt1.clear()
        self.signal_Plt1.plot(self.process.processed_G, pen=pg.mkPen('#55C355', width=2))

        self.signal_Plt2.clear()
        self.signal_Plt2.plot(self.process.processed_R, pen=pg.mkPen('#FF0000', width=2))

        self.signal_Plt3.clear()
        self.signal_Plt3.plot(self.process.processed_B, pen=pg.mkPen('#0000FF', width=2))

        self.signal_process_Plt1.clear()
        self.signal_process_Plt1.plot(self.process.samples, pen=pg.mkPen('#55C355', width=2))

        self.signal_process_Plt2.clear()
        self.signal_process_Plt2.plot(self.process.samplesPPG, pen=pg.mkPen('#FF0000', width=2))

        self.signal_process_Plt3.clear()
        self.signal_process_Plt3.plot(self.process.samplesPPG2, pen=pg.mkPen('#0000FF', width=2))

        self.signal_smooth_Plt1.clear()
        self.signal_smooth_Plt1.plot(self.process.bp, pen=pg.mkPen('#55C355', width=2))

        self.signal_smooth_Plt2.clear()
        self.signal_smooth_Plt2.plot(self.process.bpR, pen=pg.mkPen('#FF0000', width=2))

        self.signal_smooth_Plt3.clear()
        self.signal_smooth_Plt3.plot(self.process.bpB, pen=pg.mkPen('#0000FF', width=2))

        self.lblsdnn.setText("SDNN: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.sdnn)))) + " ")
        self.lblrmssd.setText("RMSSD: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.rmssd)))) + " ")

        self.lbltotaltime.setText("Total Time: \n" + ("{:.2f}".format((self.process.progress))) + " %")
        if self.process.bpms.__len__() > 50:
            if max(self.process.bpms - np.mean(self.process.bpms)) < 2:
                self.GUI_bpms = math.trunc(round(np.mean(self.process.bpms)))
                self.lblHR2.setText("Heart rate: " + "\n" + " " + str(self.GUI_bpms) + " bpm")
                self.process.GUI_bpms = self.GUI_bpms

                self.lblBP.setText("Blood Pressure: " + "\n" + " " + ("{:.2f}".format((self.process.SBP_DBP))) + '/' + (
                    "{:.2f}".format((self.process.SBP_DBP2))))

                self.lblSPO2.setText("SpO2: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.SpO2)))) + " ")

                self.lblR.setText("R: " + " " + str(float("{:.2f}".format(self.process.R))) + " ")

        if self.process.lfhf > 0:
            self.lblLF.setText("LF(nu): " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.lf)))) + " ")
            self.lblHF.setText("HF(nu): " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.hf)))) + " ")
            self.lblLFHF.setText("LF/HF: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.lfhf)))) + " ")

        if self.process.file:
            self.lbltotaltime.setText("Total Time:" + str(float("{:.2f}".format(self.process.total_time_file))) + "s\n量測完畢!感謝您!!! ")

        self.key_handler()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        if self.close_reason:
            reply = QMessageBox.critical(self, "資料錯誤 !", "提醒您 :\n請重新開啟程式，\n並請在開始偵測前，\n確認資料是否填寫完整~~~",
                                         QMessageBox.Ok, QMessageBox.Ok)
        else:
            reply = QMessageBox.information(self, "確定要關閉程式", "掰掰 !",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes or reply == QMessageBox.Ok:
            event.accept()
            self.input.stop()
            cv2.destroyAllWindows()
        else:
            event.ignore()

    def on_closing(self, event):
        self.c.closeApp.emit()

    def key_handler(self):
        self.pressed = waitKey(1) & 255  # 等待一个按键输入，waitKey(1)代表等待1ms，& 255代表取最后8位元
        if self.pressed == 27:  # 检测按下的键是否为 Esc（ASCII 27）
            #print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()

    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, 'OpenFile', r"C:\\Users\\Desktop\\", "Video files (*.mp4 *.avi *.mov)")[0]

    def openPsyDialog(self):
        self.psyDialog = QDialog()
        self.psyDialog.setWindowTitle("心理評估")
        self.psyDialog.setFixedSize(500, 500)
        self.psyDialog.setWindowModality(Qt.ApplicationModal)
        self.psyDialog.exec_()

    def saveData(self):
        # 检查输入资料是否完整，若不完整则跳出通知
        if self.input_num.text() == "" or self.input_name.text() == "" or self.input_HR.text() == "" or self.input_SBP.text() == "" or self.input_DBP.text() == "" or self.input_RR.text() == "" or self.input_Hb.text() == "" or self.input_Bilirubin.text() == "" or self.input_SPO2.text() == "":
            reply = QMessageBox.critical(self, "資料錯誤 !", "提醒您 :\n請確認資料是否填寫完整",
                                         QMessageBox.Ok, QMessageBox.Ok)
            return

        # 谵妄选项判别
        self.delirium = ["無", "明顯譫妄", "無明顯譫妄"][self.cbb_delirium.currentIndex()]

        # 抽血时间判别
        self.BloodTime = ["無", "開始抽血", "結束抽血"][self.cbb_BloodTime.currentIndex()]

        array = [
            self.input_num.text(),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            self.input_HR.text(),
            self.input_SBP.text(),
            self.input_DBP.text(),
            self.input_RR.text(),
            self.delirium,
            self.input_Hb.text(),
            self.input_Bilirubin.text(),
            self.BloodTime,
            self.input_SPO2.text(),
            self.input_Glucose.text(),
        ]
        # 檔名为feedback_yyyyMMdd
        filename = "feedback_" + time.strftime("%Y%m%d")
        saving.write_feedback_data(filename, array)

    def selectCamera(self):
        self.reset()
        self.webcam.camera = 0 if self.cbbCamera.currentIndex() == 0 else 1

    def selectROI(self):
        self.reset()
        roi_names = ["全臉", "額頭", "下巴", "臉頰"]
        roi_name = roi_names[self.cbbROI.currentIndex()]
        self.process.roi_idx = self.cbbROI.currentIndex()
        QMessageBox.warning(self, "更改掃描區域 !", "提醒您 :\n您已更改掃描區域為: " + roi_name, QMessageBox.Ok, QMessageBox.Ok)

    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")

    @QtCore.pyqtSlot()
    def main_loop(self):
        frame = self.input.get_frame()
        self.process.frame_in = frame
        self.process.run()

        self.frame = self.process.frame_out
        self.f_fr = self.process.frame_ROI
        self.bpm = self.process.bpm

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.process.GUIframe = self.frame
        cv2.putText(self.frame, "FPS " + str(float("{:.2f}".format(self.process.fps))),
                    (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                     self.frame.strides[0], QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap.fromImage(img))

        self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
        self.f_fr = np.transpose(self.f_fr, (0, 1, 2)).copy()
        f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0],
                       self.f_fr.strides[0], QImage.Format_RGB888)
        self.lblROI.setPixmap(QPixmap.fromImage(f_img))

        if self.process.sdnn >= 0:
            self.lblsdnn.setText("SDNN: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.sdnn)))) + " ")
            self.lblrmssd.setText("RMSSD: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.rmssd)))) + " ")

        self.lbltotaltime.setText("Total Time: \n" + ("{:.2f}".format((self.process.progress))) + " %")
        if len(self.process.bpms) > 50:
            if max(self.process.bpms - np.mean(self.process.bpms)) < 2:
                self.GUI_bpms = math.trunc(round(np.mean(self.process.bpms)))
                self.lblHR2.setText("Heart rate: " + "\n" + " " + str(self.GUI_bpms) + " bpm")
                self.process.GUI_bpms = self.GUI_bpms

                self.lblBP.setText("Blood Pressure: " + "\n" + " " + ("{:.2f}".format((self.process.SBP_DBP))) + '/' + (
                    "{:.2f}".format((self.process.SBP_DBP2))))

                self.lblSPO2.setText("SpO2: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.SpO2)))) + " ")

                self.lblR.setText("R: " + " " + str(float("{:.2f}".format(self.process.R))) + " ")

        if self.process.lfhf > 0:
            self.lblLF.setText("LF(nu): " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.lf)))) + " ")
            self.lblHF.setText("HF(nu): " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.hf)))) + " ")
            self.lblLFHF.setText("LF/HF: " + "\n" + " " + str(float("{:.2f}".format(np.mean(self.process.lfhf)))) + " ")

        if self.process.file:
            self.lbltotaltime.setText("Total Time:" + str(float("{:.2f}".format(self.process.total_time_file))) + "s\n量測完畢!感謝您!!! ")

        self.key_handler()

    def run(self, input):
        self.reset()
        #以下輸入
        self.process.subject_num = self.input_num.text()   #學號/編號
        self.process.subject_name = self.input_name.text() #姓名
        self.process.subject_count = self.input_count.text()

        if self.cbb_gender.currentIndex() == 1:
            self.process.subject_gender = "男"
        elif self.cbb_gender.currentIndex() == 2:
            self.process.subject_gender = "女"
        else:
            self.process.subject_gender = "unknown"

        # self.process.subject_filename = self.cbb_filename.currentIndex()
        # self.process.subject_upload = self.cbb_upload.currentIndex() == 1

        if not self.close_reason:
            input = self.input
            # self.input.dirname = self.dirname
            # if not self.input.dirname and self.input == self.video:
            #     print("choose a video first")
            #     return

            if not self.status:
                self.status = True
                input.start()
                self.btnStart.setText("Stop")
                self.cbbCamera.setEnabled(False)
                self.cbbROI.setEnabled(False)
                #self.btnOpen.setEnabled(False)

                while self.status:
                    self.main_loop()
            else:
                self.status = False
                input.stop()
                self.btnStart.setText("Start")
                self.cbbCamera.setEnabled(True)
                self.cbbROI.setEnabled(True)

