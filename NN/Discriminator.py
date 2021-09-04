import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        #Convolutional Layers
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 2) 

        #Maxpooling layers
        self.maxpooling1 = nn.MaxPool2d(2, 2)
        self.maxpooling2 = nn.MaxPool2d(2, 2)

        #Dense Layer
        self.linear1 = nn.Linear(720,100)
        self.linear2 = nn.Linear(100,1)

        #Batch normalization layer
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.batchnorm2 = nn.BatchNorm1d(100)

        

    def forward(self, x):

        # x size = [1, 28, 28]
        x = F.relu( self.batchnorm1( self.conv1(x) ) )
        x = self.maxpooling1(x)

        # x size = [ 10, 13, 13]
        x = F.relu( self.conv2(x) )
        x = self.maxpooling2(x)

        # x size = [ 20, 6, 6]
        x = x.view(x.size(0), -1)
        x = nn.Dropout(0.4)(x)

        # x size = [720]
        x = F.relu( self.batchnorm2(self.linear1(x)) )
        # x size = [100]
        x =  self.linear2(x)

        return x

'''## CNN dimension test   
dis = Discriminator()
pr = torch.randn(2,1,28,28)
res = dis(pr)
print(res.shape)'''


