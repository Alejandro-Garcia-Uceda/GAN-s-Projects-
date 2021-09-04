# Generator code
import torch
from torch import nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Convolutional Transpose layer
        self.convtr1 = nn.ConvTranspose2d(64, 64*4, kernel_size=3, stride=2)
        self.convtr2 = nn.ConvTranspose2d(64*4, 64*2, kernel_size=4, stride=1)
        self.convtr3 = nn.ConvTranspose2d(64*2, 64, kernel_size=3, stride=2)
        self.convtr4 = nn.ConvTranspose2d(64, 1,  kernel_size=4, stride=2)

        # Batch norm layer
        self.batchn1 = nn.BatchNorm2d(64*4)
        self.batchn2 = nn.BatchNorm2d(64*2)
        self.batchn3 = nn.BatchNorm2d(64)

        #Fold layer

        #Drop out layer
        self.drop1 = nn.Dropout2d(0.5)
        self.drop2 = nn.Dropout2d(0.3)

        #Activation Layer
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.tan = nn.Tanh()

    def forward(self, x):

        x = self.convtr1(x)
        x = self.batchn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.convtr2(x)
        x = self.batchn2(x)
        x = self.relu2(x)
        #x = self.drop1(x)

        x = self.convtr3(x)
        x = self.batchn3(x)
        x = self.relu3(x)
        x = self.drop2(x)

        x = self.convtr4(x)
        x = self.tan(x)

        return x

def get_noise(n_samples, device='cpu'):
    return torch.randn(n_samples, 64,1, 1, device=device)

'''#Generator Test
pr = get_noise(3)
gen = Generator()
res = gen(pr)
print(res.shape)'''