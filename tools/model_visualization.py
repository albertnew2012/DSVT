import torch
import torch.nn as nn
from torchsummary import summary

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1) # (in_channels, out_channels, kernel_size, stride)
        self.fc1 = nn.Linear(16*6*6, 10) # (in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# Create model instance
model = SimpleModel()

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a sample input tensor and move it to the same device as the model
x = torch.randn(1, 3, 8, 8).to(device)  # (batch_size, channels, height, width)

# Print the model summary
summary(model, input_size=(3, 8, 8))

from torchviz import make_dot
import torch

# Assuming you have a model and an input tensor
model = SimpleModel()
x = torch.randn(1, 3, 8, 8)  # Example input
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")