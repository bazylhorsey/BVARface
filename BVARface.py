
import train
import torch
import cv2
from torch.autograd import Variable
import numpy

class Augment(object):
    def __init__(self):
        self.augments = (cv2.imread("static/smiling.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("static/neutral.png", cv2.IMREAD_UNCHANGED),
            cv2.imread("static/tounge_out.png", cv2.IMREAD_UNCHANGED))
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.predict = self.predict_emotion_image()

    def predict_emotion_image(self, path="model.pth"):
        nn = train.NN().float()
        model = torch.load(path)
        nn.load_state_dict(model["state_dict"])
        def emotion_index(target):
            if 1 < target.shape[2]:
                target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            window = cv2.resize(target, (48,48)).reshape((1,1,48,48))
            x_axis = Variable(torch.from_numpy(window)).float()
            return numpy.argmax(nn(x_axis).data.numpy(), axis=1)[0]
        return emotion_index

    def augment_face(self, target, augment):
        target_height, target_width, _ = target.shape
        augment_height, augment_width, _ = augment.shape

        scalar = min(target_width / augment_width,
                    target_height / augment_height)

        shape = (int(augment_width * scalar), int(augment_height * scalar))
        offset = (int((target_width - shape[0]) * 0.5), int((target_height - shape[1]) * 0.5))
        scaled_target = cv2.resize(augment, shape)
        image = scaled_target[:,:,0:3]
        mask = scaled_target[:,:,3] / 255.0
        border = 1.0 - mask

        augment_target = target.copy()
        for c in range(0, 3):
            augment_target[offset[1]: offset[1] + shape[1], offset[0]: offset[0] + shape[0], c] = \
                image[:, :, c] * mask + augment_target[offset[1]: offset[1] + shape[1], offset[0]: offset[0] + shape[0], c] * border
        return augment_target

    def apply(self, img):
        window = img
        bw = cv2.equalizeHist(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(
            bw, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x0, y0, x1, y1 in faces:
            width = (x0, x0 + x1)
            height = (int(y0 - (y1 * 0.02)), int(y0 + (y1 * 0.98)))

            if width[0] < 0 or width[1] > window.shape[1] or \
                height[0] < 0 or height[1] > window.shape[0]:
                    continue

            augment = self.augments[self.predict(window[y0: y0 + y1, x0: x0 + x1])]
            window[height[0]: height[1], width[0]: width[1]] = \
                self.augment_face(window[height[0]: height[1], width[0]: width[1]], augment)
        
        return window
