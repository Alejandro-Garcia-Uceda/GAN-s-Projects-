import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        #Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 16*2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16*2, 1, kernel_size=4, stride=2) 

        #Batch normalization layer
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16*2)

        #Activate functions
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)

        #Drop out layer
        self.drop1 = nn.Dropout2d(0.5)

    def forward(self, x):

        # x size = [1, 28, 28]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.lr1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.lr2(x)

        x = self.conv3(x)

        return x.view(len(x), -1)


## CNN dimension test   
'''dis = Discriminator()
pr = torch.randn(2,1,28,28)
res = dis(pr)
print(res.shape)'''