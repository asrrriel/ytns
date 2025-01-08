import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import sys
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN
class PornDetector(nn.Module):
    def __init__(self):
        super(PornDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 31 * 31, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Amongus
torch.set_num_threads(12)

model = PornDetector().to(device)
model = torch.jit.script(model)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load():
    model.load_state_dict(torch.load('models/detect.pth',weights_only=True, map_location=torch.device(device)))


def train():
    dataset = datasets.ImageFolder(root="data", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=12)

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10000000

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    av_loss = 0
    av_loss_div = 0

    # Training Loop
    for epoch in range(num_epochs):
        for images, target in dataloader:
            # Forward pass
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            loss = criterion(outputs, target.unsqueeze(1).float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            av_loss += loss.item()
            av_loss_div += 1

        print(f"Epoch [{epoch+1}/{num_epochs}],Average Loss: {av_loss / av_loss_div:.20f}")

        #av_loss = 0
        #av_loss_div = 0

        if(epoch % 20 == 0):
            print("Autosave go brr!!!")
            torch.save(model.state_dict(), 'models/detect.pth')

    torch.save(model.state_dict(), 'models/detect.pth')

def detect(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    return output


if __name__ == '__main__':
    load()
    perc = detect( sys.argv[1] ).item() * 100
    print("Detected","Porn." if perc > 50 else "No Porn.", f"Confidence: {perc if perc > 50 else 100 - perc:.2f}%")

    #load()
    #train()
