import time

deltatime = 0

class FrameTimer():
    def __init__(self,fps = 24):
        self.last_time = time.time()
        self.FPS = fps
        self.frame_duration = 1/fps
        self.elapsed_time = 0
        self.deltatime = 0
    
    def should_update(self):
        now = time.time()
        self.deltatime = now-self.last_time
        self.elapsed_time += self.deltatime
        global deltatime 
        deltatime = self.deltatime
        self.last_time = now
        if self.elapsed_time >= self.frame_duration:
            # self.elapsed_time-=self.frame_duration
            self.elapsed_time = 0
            return True
        return False