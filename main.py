import sys
from PyQt5.QtWidgets import QApplication
from gui import GUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    while ex.status:
        ex.main_loop()
    sys.exit(app.exec_())