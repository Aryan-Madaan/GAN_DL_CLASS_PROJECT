import argparse
import math
import numpy as np
import torch
from torchvision import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from model import StyledGenerator


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample,latent_space, device):
    #arr = [latent_space.tolist() for i in range(n_sample)]
   #latent = torch.tensor(arr).to(device)
    image = generator(
        torch.tensor([latent_space.tolist()]).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images
    
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(89, 1024)
        self.hidden2= nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, 512)
        
        # Define sigmoid activation and softmax output 
        self.relu =nn.LeakyReLU()
        #self.sigmoid = nn.Tanh()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        #x = torch.mul(x,torch.tensor(7.0))
        
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=1, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=1, help='number of columns of sample matrix')
    parser.add_argument('--ckpt', type=str, help='path to latent file')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()
    
    device = 'cuda'

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()
    ckpt = torch.load(args.ckpt)
    #latent_space =  ckpt['./1/all_images/S001.jpg']['latent']
    
    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2
    test_file = np.load("trainX.npy")
    for test in range(200,300):
        latent_space = ckpt(torch.tensor(test_file[test]).to(device))
        print(latent_space)
        print(latent_space.shape)
        img = sample(generator, step, mean_style, args.n_row * args.n_col,latent_space, device)
        utils.save_image(img, f'output/sample{test}.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    #for j in range(20):
     #   img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
     #   utils.save_image(
      #      img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        #)
