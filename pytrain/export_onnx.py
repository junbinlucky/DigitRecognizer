import torch
from cnn import CNN

# Load trained pytorch model
model = torch.load('./models/model.pt', map_location=torch.device('cpu'))
# state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
# model = CNN()
# model.load_state_dict(state_dict)
model.eval()

# Create example input tensor, batch size 64, channels 1, height 28, width 28
example_input = torch.rand(1, 1, 28, 28)

# Export to onnx model
torch.onnx.export(model, example_input, './models/model.onnx')

print("Successfully export onnx model.")
