import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from cnn import CNN

############################Data loader#############################

train_data = dataset.MNIST(
    root = "mnist",
    train = True,
    transform = transforms.ToTensor(), # Transform to tensor format.
    download = True
)

test_data = dataset.MNIST(
    root = "mnist",
    train = False,
    transform = transforms.ToTensor(), # Transform to tensor format.
    download = True
)

print(train_data) # Print train dataset MNIST
print(test_data) # Print test dataset MNIST

#####################Load data in batches############################

train_loader = data_utils.DataLoader(dataset = train_data,
                                     batch_size = 64,
                                     shuffle = True)

test_loader = data_utils.DataLoader(dataset = test_data,
                                    batch_size = 64,
                                    shuffle = True)

print(train_loader) # Print train loader
print(train_loader) # Print test loader

cnn = CNN()
# Runs in cuda if have gpu device
# cnn = cnn.cuda()

########################Loss function#######################
loss_func = torch.nn.CrossEntropyLoss()

#######################Optimizer function###################
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01)

########################Train process#######################
for epoch in range(10):
    cnn.train() # Set to train mode
    for index, (images, labels) in enumerate(train_loader):
        # print(index)
        # print(images)
        # print(labels)
        # images = images.cuda()
        # labels = labels.cuda()
        # Forward propagation
        outputs = cnn(images)

        loss = loss_func(outputs, labels)

        # Clear gradient
        optimizer.zero_grad()
        # Back propagation
        loss.backward()
        # Update parameters
        optimizer.step()
        print("Current process epoch is {}, batch is {}/{}, loss is {}".format(epoch + 1,
                                                                               index,
                                                                               len(train_data)//64,
                                                                               loss.item()))

########################Test process#######################
    loss_test = 0
    rightValue = 0
    cnn.eval() # Set to eval mode
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            # images = images.cuda()
            # labels = labels.cuda()
            outputs = cnn(images)
            # print(outputs)
            # print(outputs.size())
            # print(labels)
            # print(labels.size())
            loss_test += loss_func(outputs, labels)

            _, pred = outputs.max(1)
            # print(pred)
            # Compare each element in the two tensors
            # print((pred==labels).sum().item())
            rightValue += (pred==labels).sum().item()
            print("Current process epoch is {}, batch is {}/{}, loss is {}, accuracy rate is {}".format(epoch + 1,
                                                                                                        index,
                                                                                                        len(test_data)//64,
                                                                                                        loss_test,
                                                                                                        rightValue/len(test_data)))
torch.save(cnn, "./models/model.pt")
# torch.save(cnn.state_dict(), "model.pth")