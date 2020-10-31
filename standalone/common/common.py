import time


class Timer:
    def __init__(self):
        self.init_time = time.time()

    def value(self):
        return time.time() - self.init_time
