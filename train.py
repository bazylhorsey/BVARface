import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy

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


def main():
    nn = NN().float()
    trainer = Data("fer_numpy/train.npz")
    model = {}

    load = torch.utils.data.DataLoader(trainer, shuffle=True, batch_size=32)
    loss = torch.nn.CrossEntropyLoss()
    descent = torch.optim.SGD(nn.parameters(), lr=1/1000, momentum=0.9)

    start = model.get("step", 0)
    for step in range(start, start+20):
        curr_loss = 0.
        print("loop1")
        for step, data in enumerate(load, 0):
            images = Variable(data["image"].float())
            labels = Variable(data["label"].long())
            descent.zero_grad()

            out = nn(images)
            input_loss = loss(out, labels)
            input_loss.backward()
            descent.step()

            curr_loss += input_loss.item()
            if step % 100 == 99:
                torch.save({
                    "step": step + 1,
                    "state_dict" : nn.state_dict(),
                    "optimizer": descent.state_dict()
                },
                "model.pth")
                
if __name__ == "__main__":
    main()
    print("Finished")
            

            
