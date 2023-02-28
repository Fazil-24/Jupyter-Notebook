'''
This code trains a simple convolutional neural network (CNN) on a pothole detection dataset and saves the trained model in ONNX format. Note that you'll need to replace the # Load your dataset line with code to actually load your dataset. You'll also need to modify the model architecture, loss function, and optimizer as needed for your specific use case.


pip install onnx

In this code, the ImageFolder class from torchvision.datasets is used to load the images from the training dataset directory 'path/to/training_data'. The images are transformed using the transform object defined using torchvision.transforms functions, including resizing to 32x32, converting to tensors, and normalizing the pixel values. The transformed images are then loaded into the trainloader object using torch.utils.data.DataLoader.

The training loop then trains the model on the loaded data using the defined loss function and optimizer. Finally, the trained model is converted to ONNX format using the torch.onnx.export function. This ONNX model can then be loaded and used for deployment using the Intel® Distribution of OpenVINO™ toolkit and Intel® oneAPI.
'''


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime

# Define the model architecture
class PotholeDetectionModel(nn.Module):
    def __init__(self):
        super(PotholeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 32 * 32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.fc(x)
        return x

# Initialize the model
model = PotholeDetectionModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root='path/to/training_data', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Train the model
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

# Convert the model to ONNX format
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "pothole_detection.onnx")



