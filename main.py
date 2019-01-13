"""Label images with food tags."""
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from utils import test_network, train_network
from global_vars import *
import matplotlib.pyplot as plt


class YelpDataset(data.Dataset):
    """Subclass of torch's Dataset."""

    def __init__(self, csv_file, img_folder):
        """Construct a Dataset class."""
        self.img_folder = img_folder
        imgs_labels = pd.read_csv(csv_file)
        self.images = imgs_labels["photo_id"]
        self.labels = np.array(imgs_labels.iloc[:, 1:])
        assert (len(self.images) == len(self.labels))

        # Initialize some useful img transformations
        self.transform = transforms.Compose([
            transforms.CenterCrop((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        """Define length to be the length of labels."""
        return len(self.labels)

    def __getitem__(self, index):
        """When indexing return an image and its label."""
        img = Image.open(os.path.join(
            self.img_folder, "{}.jpg".format(self.images[index])))
        return self.transform(img), torch.Tensor(self.labels[index])


class Net(nn.Module):
    """Neural network."""

    def __init__(self):
        """Construct a Net."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(54144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NO_CLASSES)

    def forward(self, x):
        """Forward propagation."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 54144)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    net = Net()
    loss_fn = nn.BCELoss()
    losses = {"steps": [], "train": [], "test": []}
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_data = YelpDataset("train.csv", "images")
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=12,
                                              shuffle=True, num_workers=4)

    test_data = YelpDataset("test.csv", "images")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=12,
                                             shuffle=True, num_workers=4)

    training_epochs = 500
    for epoch in range(training_epochs):
        print("{} of {}.".format(epoch, training_epochs))
        steps, train_loss = train_network(
            trainloader, net, loss_fn, optimizer, steps=5)
        test_loss = test_network(testloader, net, loss_fn, steps=5)
        losses["steps"].append(steps)
        losses["train"].append(train_loss)
        losses["test"].append(test_loss)

    fig = plt.figure()
    plt.plot(losses["steps"], losses["train"], label="Train loss")
    plt.plot(losses["steps"], losses["test"], label="Test loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(loc="upper right", fancybox=True)
    plt.tight_layout()
    plt.savefig("losses.pdf")
