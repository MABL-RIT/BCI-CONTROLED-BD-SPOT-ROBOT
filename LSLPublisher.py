from hashlib import new
from threading import Thread, Event
from PySide6.QtCore import QTimer, QObject, Slot, Signal

import time
from pylsl import  StreamOutlet, local_clock

class LSLPublisher(Thread, QObject):
    def __init__(self, outlet: StreamOutlet):
        Thread.__init__(self)
        
        self.outlet = outlet
        self.value = 0.0

        self.close_event = Event()

        self.value = 0

    def update_value(self, new_val: float):
        print(f"Value is  {new_val}")
        self.value = new_val

    def stop_thread(self):
        self.close_event.set()

    def run(self):
        while not self.close_event.isSet():
            stamp = local_clock()-0.125
            if self.value > 0:
                self.outlet.push_sample([self.value], stamp)
                self.value = 0
            else:
                self.outlet.push_sample([0], stamp)
                
            time.sleep(0.01)

        print('Thread is exited...')