import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=20, help='interval between image sampling')
args = parser.parse_args(args=[])
print(args)

C,H,W = args.channels, args.img_size, args.img_size

img_num = 90

img_save_path = './images-cGAN-' + str(img_num)
#img_save_path = 'images-C_dcgan'
if not os.path.exists(img_save_path):
  os.makedirs(img_save_path, exist_ok=True)
for i in range(1, args.n_epochs+1):
  for j in range(10):
    if i % 20 == 0:
      os.makedirs(img_save_path + '/epochs:' + str(i) + '/' + str(j), exist_ok=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)

'''random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True'''

class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, H*W)


    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(H*W, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)


    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input.view(input.size(0),-1)), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize Generator and discriminator
generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    print("cuda available!")
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#ImageFolderを使ってデータの取り込み
all_dataset = datasets.ImageFolder('./mnist_sampled', transform = transforms.Compose([
                                              transforms.Grayscale(1),
                                              transforms.Resize(args.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])])
)
dataloader = torch.utils.data.DataLoader(all_dataset , batch_size = args.batch_size, shuffle = True, drop_last = True)
print('the data is OK')

'''# Configure data loader
os.makedirs('../../data', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(args.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])
                   ])),
    batch_size=args.batch_size, shuffle=True, drop_last=True)
print('the data is ok')'''

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

for epoch in range(1, args.n_epochs+1):
    for i, (imgs, labels) in enumerate(dataloader):

        Batch_Size = args.batch_size
        N_Class = args.n_classes
        # Adversarial ground truths
        valid = torch.ones(Batch_Size).cuda()
        fake = torch.zeros(Batch_Size).cuda()

        # Configure input
        real_imgs = imgs.type(torch.FloatTensor).cuda()

        real_y = torch.zeros(Batch_Size, N_Class)
        real_y = real_y.scatter_(1, labels.view(Batch_Size, 1), 1).cuda()
        #y = Variable(y.cuda())

        # Sample noise and labels as generator input
        noise = torch.randn((Batch_Size, args.latent_dim)).cuda()
        gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
        gen_y = torch.zeros(Batch_Size, N_Class)
        gen_y = gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss for real images
        d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
        # Loss for fake images
        gen_imgs = generator(noise, gen_y)
        d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        #gen_imgs = generator(noise, gen_y)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs,gen_y).squeeze(), valid)

        g_loss.backward()
        optimizer_G.step()


        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                            d_loss.data.cpu(), g_loss.data.cpu()))

        if epoch % 20 == 0:
          for j in range(img_num):
            noise = torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, args.latent_dim)         save_gen_imgs = gen_imgs[i*10]
              save_image(save_gen_imgs, img_save_path + '/epochs:%d/%d/epoch:%d-%d_%d.png' % (epoch, i, epoch,i, j), normalize=True)

print('finish!')
