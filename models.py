## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (64, 106, 106)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (128, 49, 49)
        # after one pool layer, this becomes (128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        
        ## output size = (W-F)/S +1 = (24-5)/1 +1 = 20
        # the output Tensor for one image, will have the dimensions: (128, 20, 20)
        # after one pool layer, this becomes (256, 10, 10)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.bn4 = nn.BatchNorm2d(256)
        
        ## output size = (W-F)/S +1 = (10-5)/1 +1 = 6
        # the output Tensor for one image, will have the dimensions: (128, 6, 6)
        # after one pool layer, this becomes (512, 3, 3)
        self.conv5 = nn.Conv2d(256, 512, 5)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Max pooling layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # 256 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(512*3*3, 4096)
                
        # Fully connected layers with 4096 inputs and 1024
        self.fc2 = nn.Linear(4096, 1024)
        
        # Create 136 output channels (for the 2*86 keypoints)
        self.fc3 = nn.Linear(1024, 136)
        
        # dropout with p=0.3
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # five conv/relu + pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
