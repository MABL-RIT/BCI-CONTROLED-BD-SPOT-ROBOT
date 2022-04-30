import logging, sys
from PySide6.QtWidgets import QApplication
from mainwindow import MainWindow

'''
To make changes in ui, use qt designer and save the changes in the ui file.
To generate ui code run the code: pyside2-uic app/mainwindow.ui -o app/ui_mainwindow.py
'''

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # with open("config.yaml", 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # print(config)

    app = QApplication(sys.argv)

    w = MainWindow()

    app.exec()