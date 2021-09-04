import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from torchsummary import summary

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datetime import timedelta
import time

from NN import Discriminator2 as D 
from NN import Generator2 as G 


#### Function to save the images
def show_tensor_images(image_tensor, fake, num_images=25, size=(1, 28, 28)):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    if num_images >= 5:
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
    else:
        image_grid = make_grid(image_unflat[:num_images], nrow=num_images)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    if fake == True:
        ti = r'graphs\examples_step{0}.jpg'.format(str(epoch)) 
    else:
        ti = r'graphs\real_numbers.jpg' 

    plt.savefig(ti)
    plt.close()

####### Function to save the graphs of the model evolution 
def show_graph(DISC_LOSS, GEN_LOSS,  mem_allo, mem_res, epoch):
    figure_1, (ax1, ax2) = plt.subplots(2)

    ax1.plot(range(len(DISC_LOSS)),DISC_LOSS, label="Discriminator Loss")
    ax1.plot(range(len(GEN_LOSS)),GEN_LOSS, label="Generator Loss")
    ax1.legend()

    # print(DISC_LOSS)
    # print(GEN_LOSS)
    ax2.plot(range(len(DISC_LOSS)),np.array(DISC_LOSS)/np.array(GEN_LOSS), label="Gen and Disc loss relation")
    ax2.legend()

    #plt.title('Loss value step' + str(epoch))
    plt.savefig(r'graphs\loss_values.jpg')
    plt.close()
    
    plt.plot(range(len(mem_allo)), mem_allo, label="Allocate memory")
    plt.plot(range(len(mem_res)), mem_res, label="Reserved memory")
    plt.legend()
    plt.title('CUDA Memory' + str(epoch))
    plt.savefig(r'graphs\Cuda_memory.jpg')
    plt.close()

###### 1. Import and preprocessing of images ########
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# It is not a conditional GAN, for this reason we dont need Y_train and Y_test
del(Y_train, Y_test)
## X_train have 60,000 images, I will use it to train the GAN
# Transform images: normalize and transform to torch 
images_tr = []
for i in X_train:
    i = np.array(i, dtype=np.float64)/255 # Normalize, max value 1, min value 0 
    images_tr.append( torch.tensor([i]) )

images_te = []
for i in X_test:
    i = np.array(i, dtype=np.float64)/255 # Normalize, max value 1, min value 0 
    images_te.append( torch.tensor([i]) )
## The image size is [1, 28, 28]


#criterion = nn.BCEWithLogitsLoss()
criterion = nn.SmoothL1Loss()
#z_dim = 64
display_step = 500
batch_size = 45
lrd = 0.0001
lrg = 0.0001
n_epochs = 200

beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# Transform the list with the images to Dataloder
dataloader = DataLoader( images_tr, batch_size=batch_size, shuffle=True)

print('Dataloader ok')

#Import the generator and dicriominator
#INICIALICE THE GENERATOR AND DISCRIMINATOR
gen = G.Generator().to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lrg, betas=(beta_1, beta_2))
disc = D.Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lrd, betas=(beta_1, beta_2))
print('Generator and Clasificator imported')

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

#TRAINING
cur_step = 0 #count batchs
mean_generator_loss = 0 
mean_discriminator_loss = 0

GEN_LOSS = []
DISC_LOSS = []

i =1; j = 1

# Memori estate
mem_allo = [] # Allocated memory cuda
mem_res = [] # reserved memory cuda

#nloss = 20
now = time.time()
epoch = 1

for epoch in range(n_epochs):
    # Dataloader returns the batches
    timestamp = time.time() - now
    print("Tiempo ejecución:", timedelta(seconds=timestamp), 'Epoca:', epoch+1, 'de', n_epochs,'  ', 100 * (epoch+1)/n_epochs, '%', ' ', timedelta(seconds=timestamp/(epoch+1)*(n_epochs-epoch+1)))

    for real in tqdm(dataloader):

        mem_allo.append(torch.cuda.memory_allocated())
        mem_res.append(torch.cuda.memory_reserved())
        cur_batch_size = len(real)
        #print(cur_batch_size)
        real = real.to(device)

        ## Update discriminator ##
        for _ in range(i):

            disc_opt.zero_grad()
            fake_noise = G.get_noise(cur_batch_size, device=device) #[128, 64]
            fake = gen(fake_noise) #[128, 1, 28, 28]
            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            
            disc_real_pred = disc(real.float())
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            #disc_loss = (disc_fake_loss*0.6 + 0.4*disc_real_loss) 

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()
            # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / (display_step)

        ## Update generator ##
        for _ in range(j):

            gen_opt.zero_grad()
            fake_noise = G.get_noise(batch_size, device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake)

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            GEN_LOSS.append(mean_generator_loss)
            DISC_LOSS.append(mean_discriminator_loss)
            
            mean_generator_loss = 0
            mean_discriminator_loss = 0

            try:
                show_graph(DISC_LOSS, GEN_LOSS, mem_allo, mem_res, epoch = epoch)
            except:
                continue
        cur_step += 1
    
    
    if epoch % 10 == 0 and epoch > 0:
        show_tensor_images(fake, fake = True)

        if epoch == 10:
            show_tensor_images(real, fake = False)

show_graph(DISC_LOSS, GEN_LOSS, mem_allo, mem_res, epoch = epoch)
show_tensor_images(fake, fake = True)
torch.save(gen, 'generator1.pth')
