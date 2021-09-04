# Generator code
import torch
from torch import nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Dense layers
        self.linear = nn.Linear(10, 36)

        # Convolutional Transpose layer
        self.convtr1 = nn.ConvTranspose1d(1, 10, 3, 2)
        self.convtr2 = nn.ConvTranspose2d(1, 30, 2, 1)
        self.convtr3 = nn.ConvTranspose2d(30, 10, 2, 2)
        self.convtr4 = nn.ConvTranspose2d(10, 1, 2, 2)


        # Batch norm layer
        self.batchn1 = nn.BatchNorm2d(30)
        self.batchn2 = nn.BatchNorm2d(10)

        #Fold layer
        #self.fold = nn.Fold(

        #Drop out layer
        self.drop1 = nn.Dropout2d(0.5)

        #Activation Layer
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        


    def forward(self, x):
        
        # x size = [1, 10]
        x = self.linear(x)

        # x size = [1, 10]
        #x = self.convtr1(x)
        x = x.view(x.size(0), 1, 6, 6)

        # x size = [1, 6, 6]
        x = self.batchn1( self.convtr2(x) )
        x = self.relu1(x)
        x = self.drop1(x)

        # x size = [30, 7, 7]
        x = self.batchn2( self.convtr3(x) )
        x = self.relu2(x)
        #x = self.convtr4(x)

        # x size = [1, 28, 28]
        return x

def get_noise(n_samples, device='cpu'):
    return torch.randn(n_samples, 1, 10, device=device)

'''#Generator Test
pr = get_noise(3)
gen = Generator()
res = gen(pr)
print(res.shape)'''
