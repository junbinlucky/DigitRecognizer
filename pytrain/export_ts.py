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

# Convert to torchscript via tracing
ts_model = torch.jit.trace(model, example_input)

# Save the torchscript model
ts_model.save("./models/model.ts.pt")

print("Successfully converted to torchscript and saved as torchscript model.")
