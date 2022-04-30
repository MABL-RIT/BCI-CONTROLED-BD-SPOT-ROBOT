from PySide6.QtWidgets import QMainWindow, QDialog
from PySide6.QtCore import QTimer, Slot, Signal, Qt

from ui_wait_form import Ui_Dialog

class WaitForm(QDialog, Ui_Dialog):
    closed = Signal()
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # self.setWindowState(Qt.WindowMaximized)

        QTimer.singleShot(3000, self.close)

        self.show()

    def set_label(self, label: str):
        self.label.setText(label)
    
    def closeEvent(self, event):
        event.accept()
        self.closed.emit()
        self.close()
    
      
