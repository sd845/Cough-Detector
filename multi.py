from threading import Thread
import cv2

class WebcamVideoWriter(object):
    def __init__(self, src=0,videoname = 0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc(*'MJPG')
        self.output_video = cv2.VideoWriter(videoname, self.codec, 1.9, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(10)
        if not self.status:
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)


    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

    def get_video(self,filename,videoname):
        # webcam_videowriter = WebcamVideoWriter(filename,videoname)
        while True:
            try:
                self.show_frame()
                self.save_frame()
            except AttributeError:
                pass
        self.capture.release()
        self.output_video.release()
        cv2.destroyAllWindows()
        exit(1)
