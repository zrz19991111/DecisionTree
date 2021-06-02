import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

BATCH_SIZE = 16#每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else"cpu")
EPOCHS = 10

pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1031,),(0.3081,))
])

from torch.utils.data import DataLoader

train_set = datasets.MNIST("data",train=True,download=True,transform=pipeline)
test_set = datasets.MNIST("data",train=False,download=True,transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True)

with open("./data/MNIST/raw/train-image-idx3-ubyte","rb") as f:
    file = f.read()

    image1 = [int(str(item).encode('ascii'), 16) for item in file[16: 16 + 784]]
    print(image1)

    import cv2
    import numpy as np

    image1_np = np.array(image1, dtype=np.unit8).reshape(28, 28, 1)

    print(image1.np.shape)

    cv2.imwrite("digit.jpg", image1.np)


    class Digit(nn.Moudle):
        def_init_(self)
        super()._init_()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc1 = nn.Linear(500, 10)


    def forward(self, x)
        input_size = x.size(0)
        x = self.conv1(x)
        x=F.relu(x)
        F = .max_pool2d(x,2,2)

        x=self.conv2(x)
        x=F.relu(x)

        x=x.view(input_size, -1)
        x=self.fc1(x)
        x=F.relu(x)

        x=self.fc2(x)

        output=F.log_softmax(x,dim=1)

        return  output

