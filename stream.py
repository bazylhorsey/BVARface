import threading
import base64
import binascii
import time
import cv2
import numpy

def encode(img):
    _, arr = cv2.imencode(".jpg", img)
    return base64.b64encode(arr.tobytes())


def decode(img):
    img_bytes = base64.b64decode(img)
    arr = numpy.frombuffer(img_bytes, dtype=numpy.uint8)
    return cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)

class Video(object):
    def __init__(self, augment):
        self.request = []
        self.response = []
        self.augment = augment

        thread = threading.Thread(target=self.spin, args=())
        thread.daemon = True
        thread.start()

    def produce_output_frame(self):
        if not self.request:
            return

        inframe = decode(self.request.pop(0))

        outframe = encode(self.augment.apply(inframe))

        self.response.append(binascii.a2b_base64(outframe))

    def spin(self):
        while True:
            self.produce_output_frame()
            time.sleep(0.01)

    def enqueue(self, img):
        self.request.append(img)

    def dequeue(self):
        while not self.response:
            time.sleep(0.05)
        return self.response.pop(0)



