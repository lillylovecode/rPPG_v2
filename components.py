from PyQt5.QtWidgets import QLabel, QLineEdit, QComboBox, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject,pyqtSignal
import pyqtgraph as pg

class Communicate(QObject):
    closeApp = pyqtSignal()

def create_label(parent, text, x, y, width, height, font):
    label = QLabel(parent)
    label.setGeometry(x, y, width, height)
    label.setFont(font)
    label.setText(text)
    return label


def create_input_field(parent, label_text, x, y, placeholder, obj_name, font, label_font):
    create_label(parent, label_text, x, y, 60, 60, label_font)
    input_field = QLineEdit(parent)
    input_field.move(x + 90, y + 10)
    input_field.setFixedWidth(170)
    input_field.setFixedHeight(45)
    input_field.setFont(font)
    input_field.setPlaceholderText(placeholder)
    input_field.setObjectName(obj_name)
    return input_field


def create_combobox(parent, label_text, x, y, items, font):
    create_label(parent, label_text, x, y, 100, 60, font)
    combobox = QComboBox(parent)
    combobox.move(x + 130, y + 10)
    for item in items:
        combobox.addItem(item)
    combobox.setFixedWidth(170)
    combobox.setFixedHeight(45)
    combobox.setFont(font)
    return combobox

def create_plot(parent, x, y, width, height, background, label):
    plot = pg.PlotWidget(parent)
    plot.move(x, y)
    plot.resize(width, height)
    plot.setBackground(background)
    plot.setLabel('bottom', label)
    return plot


def create_basic_info_input(parent, placeholder, x, y):
    input_field = QLineEdit(parent)
    input_field.move(x, y)
    input_field.setFixedWidth(170)
    input_field.setFixedHeight(45)
    input_field.setFont(QFont())
    input_field.setPlaceholderText(placeholder)
    return input_field


def create_basic_info_combobox(parent, items, x, y, callback=None):
    combobox = QComboBox(parent)
    combobox.move(x, y)
    for item in items:
        combobox.addItem(item)
    combobox.setFixedWidth(170)
    combobox.setFixedHeight(45)
    combobox.setFont(QFont())
    if callback:
        combobox.currentIndexChanged.connect(callback)
    return combobox

def show_warning_message():
    lblWarning1 = " ~~~給實驗人員的小提醒~~~\n"
    lblWarning2 = "  *1.檢查資料抓幾筆\n"
    lblWarning3 = "  *2.注意雲端和Excel是否關閉\n"
    lblWarning4 = "  *3.幫忙測量資料並協助輸入\n"
    lblWarning5 = "  *4.建檔選項需注意是否選擇正確"
    lblWarning = lblWarning1 + lblWarning2 + lblWarning3 + lblWarning4 + lblWarning5
    QMessageBox.warning(None, "~小提醒~", lblWarning, QMessageBox.Ok, QMessageBox.Ok)

