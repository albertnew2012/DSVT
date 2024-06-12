import torch
import torch.nn as nn

# given a deep learning, how to print out its weigths
model = None
for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")  # Adjust as needed to print more/less values
