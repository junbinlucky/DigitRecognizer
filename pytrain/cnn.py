import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential( # Sequence container
            # 1.Convolution operation
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), # h2 = (28-5+2*2)/1
            # 2.Normalized the BN layer
            torch.nn.BatchNorm2d(num_features=32),
            # 3.Activate layer
            torch.nn.ReLU(),
            # 4.Max pooling
            torch.nn.MaxPool2d(2) # max pooling kernel size 2*2
        )
        # Full connection layer
        self.fc = torch.nn.Linear(in_features=14*14*32, out_features=10)
    
    def forward(self, x):
        out = self.conv(x)
        # Expand image into one dimension, input tensor x is (n, c, h, w) with four dimension.
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
