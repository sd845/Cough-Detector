import threading
import binascii
import base64
from time import sleep
import io
from PIL import Image
import cv2
import numpy as np


class Camera(object):
    def __init__(self, makeup_artist):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist
        self.labelname = ""
        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string.
        input_str = self.to_process.pop(0)
        label = self.labelname

        # convert it to a pil image
        # input_img = self.base64_to_image(input_str)

        ################## where the hard work is done ############
        # output_img is an PIL image
        output_img = self.makeup_artist.apply_makeup(label,input_str)

        # output_str is a base64 string in ascii
        # output_str = self.image_to_base64(output_img)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(output_img)

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input,label):
        self.to_process.append(input)
        self.labelname = label


    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)

    def image_to_base64(self,file):
        """imgByteArr = io.BytesIO()
        file.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        msg = base64.b64encode(imgByteArr)
        return msg"""
        retval, buffer = cv2.imencode('.jpg', file)
        msg = base64.b64encode(buffer)

    def base64_to_image(self,msg):
        msg = base64.b64decode(msg)
        """buf = io.BytesIO(msg)
        img = Image.open(buf)
        return img"""
        nparr = np.fromstring(msg, np.uint8)
        retval,img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[1]
        return img
