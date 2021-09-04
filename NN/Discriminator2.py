import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        #Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 16*2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16*2, 16*4, kernel_size=4, stride=2) 

        #Maxpooling layers
        self.maxpooling1 = nn.MaxPool2d(2, 2)

        #Dense Layer
        self.linear1 = nn.Linear(128,80)
        self.linear2 = nn.Linear(80,1)


        #Batch normalization layer
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16*2)
        self.batchnorm3 = nn.BatchNorm1d(80)

        #Activate functions
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)

        #Drop out layer
        self.drop1 = nn.Dropout2d(0.5)
        self.drop1d = nn.Dropout(0.5)

        #Sigmod
        self.sigm = nn.Sigmoid()


    def forward(self, x):

        # x size = [1, 28, 28]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.lr1(x)
        x = self.maxpooling1(x)
        x = self.drop1(x)
       
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.lr2(x)

        #x = self.conv3(x)
        x = x.view(len(x), -1)

        x = self.linear1(x)
        x = F.relu( self.batchnorm3(x) )
        x = self.drop1d(x)
        x = self.linear2(x)

        #x = self.sigm(x)




        return x


## CNN dimension test   
'''dis = Discriminator()
pr = torch.randn(2,1,28,28)
res = dis(pr)
print(res.shape)'''