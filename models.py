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
        
        #layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 192, 2)
        self.conv5 = nn.Conv2d(192, 320, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.15)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.3)
        self.dropout6 = nn.Dropout(0.35)
        self.dropout7 = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(11520,1000)
        self.dense2 = nn.Linear(1000,1000)
        self.dense3 = nn.Linear(1000,136)
        
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        
        #initialize weights
        nn.init.uniform_(self.conv1.weight,b=.01)
        nn.init.uniform_(self.conv2.weight,b=.01)
        nn.init.uniform_(self.conv3.weight,b=.01)
        nn.init.uniform_(self.conv4.weight,b=.01)
        nn.init.uniform_(self.conv5.weight,b=.01)
        nn.init.xavier_normal_(self.dense1.weight,gain=.7)
        nn.init.xavier_normal_(self.dense2.weight,gain=.7)
        nn.init.xavier_normal_(self.dense3.weight,gain=.7)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.elu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        x = self.elu(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.pool(x)
        x = self.elu(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.pool(x)
        x = self.elu(x)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.pool(x)
        x = self.elu(x)
        x = self.dropout5(x)
        
        #x = x.reshape(x.size(0),-1).squeeze()
        x = x.view(x.size(0), -1)
        
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dropout6(x)
        
        x = self.dense2(x)
        x = self.elu(x)
        x = self.dropout7(x)
        
        x = self.dense3(x)
        #x = self.tanh(x)*2.5
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
