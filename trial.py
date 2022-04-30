import time
from LSLPublisher import LSLPublisher
from PySide6.QtMultimedia import QMediaPlayer
from pylsl import  StreamOutlet
import random
from wait_form import WaitForm

class Trial:
    player : QMediaPlayer
    lsl_pub : LSLPublisher
    
    def __init__(self, player, outlet: StreamOutlet, num_repeat: int) -> None:
        self.player = player
        self._outlet = outlet
        self.player.mediaStatusChanged.connect(self.video_end)

        self.video_list = {
            'forward'   : { 'path': 'video_folder/Forward.mp4', 'id': 1 },
            'backward'  : { 'path': 'video_folder/Reverse.mp4', 'id': 2 },
            'left'      : { 'path': 'video_folder/Left.mp4', 'id': 3 },
            'right'     : { 'path': 'video_folder/Right.mp4', 'id': 4 },
            'turn right': { 'path': 'video_folder/RotateRight.mp4', 'id': 5 },
            'turn left' : { 'path': 'video_folder/RotateLeft.mp4', 'id': 6 },
            'stand'     : { 'path': 'video_folder/Stand.mp4', 'id': 7 },
            'sit'       : { 'path': 'video_folder/Sit.mp4', 'id': 8 },
            'nothing'   : { 'path': 'video_folder/Nothing.mp4', 'id': 10 },
        }

        self.video_squence = list(self.video_list.keys()) * num_repeat
        random.shuffle(self.video_squence)
    
        self.video_list['transition'] = { 'path': 'video_folder/GreenScreen.mp4', 'id' : 9}

        print(f"It is going to take {(len(self.video_squence)*18)/60} mins")

        temp = []
        for video in self.video_squence:
            temp.append(video)
            temp.append('transition')
        
        self.video_squence = temp

        self.lsl_pub = LSLPublisher(self._outlet)

        self.video_counter = 0

    def stop_trial(self):
        self.lsl_pub.stop_thread()

    def start_trial(self):
        # lsl start
        self.lsl_pub.start()

        self.play(self.video_counter)
        
    def play(self, cnt: int):
        name = self.video_squence[self.video_counter]
        if name != "transition":
            wait = WaitForm()
            wait.set_label(label=name.capitalize())
            wait.exec()
        
        path = self.video_list[self.video_squence[cnt]]['path']
        id = self.video_list[self.video_squence[cnt]]['id']
        self.player.setSource(path)

        self.lsl_pub.update_value(id)
        self.player.play()

    def video_end(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            name = self.video_squence[self.video_counter]
            print(f'Video end name {name}, counter: {self.video_counter}')
            self.video_counter += 1
            
            if self.video_counter < len(self.video_squence):
                self.play(self.video_counter)
                return

            self.lsl_pub.stop_thread()
    