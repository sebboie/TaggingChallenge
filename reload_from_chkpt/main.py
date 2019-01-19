"""Label images with food tags."""
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image


NO_CLASSES = 38
HEIGHT = 200
WIDTH = 300


class DataTransformer:
    """Subclass of torch's Dataset."""

    def __init__(self):
        """Construct a DataTransformer class that contains transformations."""
        self.transformations = transforms.Compose([
            transforms.CenterCrop((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def transform(self, img):
        """Apply transformation."""
        img = self.transformations(img)

        # If passing a single image, add a singleton dimension in the front
        # (batchsize of 1).
        if len(img.shape) == 3:
            img.unsqueeze_(0)

        assert len(img.shape) == 4, "The image data does not have 4 "
        "dimensions. But it should!"

        return img


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
    chkpt_file = "checkpoint.pt"
    img_file = "l6nlHc_jUa2aXQU_6SNqUA.jpg"
    test_data = pd.read_csv("test.csv")

    net = Net()
    checkpoint = torch.load(chkpt_file)
    net.load_state_dict(checkpoint["model_state_dict"])
    transformer = DataTransformer()

    # Instantiate a DataFrame that contains the correct labels
    label = test_data[test_data["photo_id"] == img_file.split(".jpg")[0]]
    label = label.drop(columns="photo_id")

    # Open the image and apply transformations
    image = Image.open(img_file)
    image = transformer.transform(image)

    # We are not training, hence we don't need to store gradients
    with torch.no_grad():
        y_hat = net.forward(image)

    # Round the predictions and create a DataFrame to compare ground truth
    # and predictions
    y_hat = np.round(y_hat.numpy().squeeze()).astype(int)
    prediction = pd.DataFrame({
        lab: [val] for lab, val in zip(label.columns, y_hat)
    })
    result = pd.concat([label, prediction], ignore_index=True)
    result = result.rename({0: "label", 1: "prediction"}, axis="index")
    print(result.T)
