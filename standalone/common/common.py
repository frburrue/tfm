import time
import enum


class Timer:
    def __init__(self):
        self.init_time = time.time()

    def value(self):
        return time.time() - self.init_time


class OutputFilter(enum.Enum):

    APP_IN = 0
    DETECTION = 1
    REKOGNITION = 2
    PROCESSING = 3
    APP_OUT = 4

