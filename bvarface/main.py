import cv2
import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional
import torch.nn
import torch.optim
from torch.utils.data import Dataset

class Data(Dataset):
  def __init__(self, path):
    with numpy.load(path) as data:
      self.samples = data["x_list"]
      self.labels = data["y_list"]

    self.samples = self.samples.reshape((-1, 1, 48, 48))
    self.x_axis = Variable(torch.from_numpy(self.samples)).float()
    self.y_axis = Variable(torch.from_numpy(self.labels)).float()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    return {
      "image": self.samples[i],
      "label": self.labels[i]
    }

trainer = Data("bvarface/fer_numpy/train.npz")
tester = Data("bvarface/fer_numpy/test.npz")

spin_train = torch.utils.data.DataLoader(trainer, batch_size=32, shuffle=True)
spin_test = torch.utils.data.DataLoader(trainer, batch_size=32, shuffle=False)

class NN(torch.nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.pool = torch.nn.MaxPool2d(2, 2)

    self.conv1 = torch.nn.Conv2d(1, 6, 5)
    self.conv2 = torch.nn.Conv2d(6, 6, 3)
    self.conv3 = torch.nn.Conv2d(6, 16, 3)

    self.fc1 = torch.nn.Linear(256, 120)
    self.fc2 = torch.nn.Linear(120, 48)
    self.fc3 = torch.nn.Linear(48,3)

  def forward(self, step):
    step = self.pool(torch.nn.functional.relu(self.conv1(step)))
    step = self.pool(torch.nn.functional.relu(self.conv2(step)))
    step = self.pool(torch.nn.functional.relu(self.conv3(step)))
    step = step.view(-1, 256)
    step = torch.nn.functional.relu(self.fc1(step))
    step = torch.nn.functional.relu(self.fc2(step))
    return self.fc3(step)


def predict_emotion_image(path="bvarface/fer_dataset/model.pth"):
  nn = NN().float()
  model = torch.load(path)
  nn.load_state_dict(model["state_dict"])
  def emotion_index(target):
    if 1 < target.shape[2]:
      target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    window = cv2.resize(target, (48,48)).reshape((1,1,48,48))
    x_axis = Variable(torch.from_numpy(window)).float()
    return numpy.argmax(nn(x_axis).data.numpy(), axis=1)[0]
  return emotion_index

def augment_face(target, augment):
    target_height, target_width, _ = target.shape
    augment_height, augment_width, _ = augment.shape
    augment_copy = augment

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

def main():
    augments = (cv2.imread("bvarface/static/smiling.png", cv2.IMREAD_UNCHANGED),
      cv2.imread("bvarface/static/neutral.png", cv2.IMREAD_UNCHANGED),
      cv2.imread("bvarface/static/tounge_out.png", cv2.IMREAD_UNCHANGED))
    cap = cv2.VideoCapture(0)


    predict_emotion = predict_emotion_image()
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        _, window = cap.read()
        bw = cv2.equalizeHist(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))

        faces = cascade.detectMultiScale(
            bw, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x0, y0, x1, y1 in faces:
            width = (x0, x0 + x1)
            height = (int(y0 - (y1 * 0.02)), int(y0 + (y1 * 0.98)))

            if width[0] < 0 or width[1] > window.shape[1] or \
                height[0] < 0 or height[1] > window.shape[0]:
                    continue

            augment = augments[predict_emotion(window[y0: y0 + y1, x0: x0 + x1])]
            window[height[0]: height[1], width[0]: width[1]] = \
                augment_face(window[height[0]: height[1], width[0]: width[1]], augment)


        cv2.imshow("BVAR Face", window)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

