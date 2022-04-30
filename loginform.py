from PySide6.QtWidgets import QMainWindow, QWidget
from PySide6.QtCore import QTimer, Slot, Signal

from ui_login import Ui_Form

class LoginForm(QWidget, Ui_Form):
    closed = Signal()
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.show()

        self.pushButtonStart.clicked.connect(self.close)
        self.repeated_value = 3
        self.spinBoxRepeatCount.valueChanged.connect(self.update_value)

    def repeat_val(self):
        return self.repeated_value

    def update_value(self, val):
        self.repeated_value = val

    def closeEvent(self, event):
        event.accept()
        self.closed.emit()
        self.close()
    
      
