import sys
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from trial import Trial
from ui_mainwindow import Ui_MainWindow
from loginform import LoginForm
from pylsl import StreamInfo, StreamOutlet, local_clock

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setupUi(self)

        self.outlet = self.create_lsl()
        
        self.login = LoginForm()
        self.login.show()


        self.login.closed.connect(self.start_main_screen)

        self.setWindowState(Qt.WindowMaximized)

        self._player = QMediaPlayer()

        self._video_widget = QVideoWidget()
        self.setCentralWidget(self._video_widget)
        self._player.setVideoOutput(self._video_widget)

    def create_lsl(self):
        
        info = StreamInfo('STI001', 'Events', 1, 100, 'float32', '6448269')
        info.desc().append_child_value("manufacturer", "MABL")
        channels = info.desc().append_child("channels")
        for c in ["STI"]:
            channels.append_child("channel")\
                .append_child_value("name", c)\
                .append_child_value("unit", "event")\

        return StreamOutlet(info)

    def start_main_screen(self):
        num_repeat = self.login.repeat_val()
        print(num_repeat)
        self.trial = Trial(self._player, self.outlet, num_repeat)
        self.show()
        
        self.trial.start_trial()

    def closeEvent(self, event):
        if self._player.playbackState() != QMediaPlayer.StoppedState:
            self._player.stop()

        self.trial.stop_trial()
        event.accept()
        self.close()

