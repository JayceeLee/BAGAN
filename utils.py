from easydict import EasyDict as edict
from PIL import Image
from collections import OrderedDict
import yaml
import imageio
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import math

def conditional_latenent_generat(distribution, class_num, gen_num):	# gen_num : the number of fake images per batch
	gen_each = math.ceil(gen_num/class_num)
	fake_z = distribution[0].sample((gen_each,))
	label_z = torch.zeros(gen_each, dtype=torch.long)
	tmp = label_z
	for i in range(1, class_num):
		fake_z = torch.cat((fake_z, distribution[i].sample((gen_each,))), dim=0)	# [fake_each, z_dim]
		label_z = torch.cat((label_z, tmp.fill_(i)), dim=0)	# because just 1-dim
	#print("fake_z size : {}".format(fake_z.shape))
	#print("label_z size : {}".format(label_z.shape))
	#print(label_z)
	return fake_z, label_z
	

def batch2one(Z, y, z, class_num):
	for i in range(y.shape[0]):
		Z[y[i]] = torch.cat((Z[y[i]], z[i].cpu()), dim=0) # Z[label][0] should be deleted..
	return Z			
	
class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def one_hot(x, num_classes):
        '''
        One-hot encoding of the vector of classes. It uses number of classes + 1 to
        encode fake images
        :param x: vector of output classes to one-hot encode
        :return: one-hot encoded version of the input vector
        '''
        label_numpy = x.data.cpu().numpy()
        label_onehot = np.zeros((label_numpy.shape[0], num_classes + 1))
        label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
        return torch.FloatTensor(label_onehot)


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        print("Found {} images in subfolders of: {}".format(len(imgs), root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self):
        return len(self.imgs)

def load_data(train_dir, transform, data_name, config):
        if 'mnist' in data_name:
                global dataset_size
                dataset_size = datasets.MNIST(train_dir, True, transform, download=True).__len__()
                print("total : {}".format(dataset_size))
                return torch.utils.data.DataLoader(datasets.MNIST(train_dir, True, transform, download=True), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=False)
        elif 'celebA' in data_name:
                return torch.utils.data.DataLoader(ImageFolder(train_dir, transform), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=False)
        elif 'cifar10' in data_name:
                return torch.utils.data.DataLoader(datasets.CIFAR10(train_dir, True, transform, download=True), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=False)
        elif 'test' in data_name:
                return torch.utils.data.DataLoader(ImageFolder(train_dir, transform), batch_size=1, shuffle=False, num_workers=config.workers, pin_memory=False)
        else:
                return

def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')

def print_gan_log(epoch, epoches, iteration, iters, learning_rate,
              display, batch_time, data_time, D_losses, G_losses):
    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
          'Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'
          'Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'.format(
              display, batch_time=batch_time,
              data_time=data_time, loss_D=D_losses, loss_G=G_losses))

def print_vae_log(epoch, epoches, iteration, iters, learning_rate,
              display, batch_time, data_time, losses):

    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
          'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
              display, batch_time=batch_time,
              data_time=data_time, loss=losses))


def plot_result2(fake, image_size, num_epoch, save_dir, name, fig_size=(8, 8), is_gray=False):

    generate_images = fake
    #G.train() # for next train after plot_result at a epoch ...

    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    if name == "dcgan":
        plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
        plt.close()

    elif name == "anomaly":
        plt.savefig(os.path.join(save_dir, 'anoGAN_epoch_{}.png'.format(num_epoch)))
        plt.close()

    elif name == "vae":
        plt.savefig(os.path.join(save_dir, 'vae_epoch_{}.png'.format(num_epoch)))
        plt.close()

    elif name =="ssgan":
        plt.savefig(os.path.join(save_dir, 'ssgan_epoch_{}.png'.format(num_epoch)))
        plt.close()

def plot_result(G, fixed_noise, image_size, num_epoch, save_dir, name, fig_size=(8, 8), is_gray=False):

    G.eval()
    generate_images = G(fixed_noise)
    G.train() # for next train after plot_result at a epoch ... 
    
    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    
    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')
    
    if name == "dcgan":
        plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
        plt.close()

    elif name == "anomaly":
        plt.savefig(os.path.join(save_dir, 'anoGAN_epoch_{}.png'.format(num_epoch)))
        plt.close()

    elif name == "vae":
        plt.savefig(os.path.join(save_dir, 'vae_epoch_{}.png'.format(num_epoch)))
        plt.close()
    
    elif name =="ssgan":
        plt.savefig(os.path.join(save_dir, 'ssgan_epoch_{}.png'.format(num_epoch)))
        plt.close()

    
def plot_loss(num_epoch, epoches, save_dir, **loss):
    fig, ax = plt.subplots() 
    ax.set_xlim(0,epoches + 1)
    if len(loss) == 2:
        ax.set_ylim(0, max(np.max(loss['g_loss']), np.max(loss['d_loss'])) * 1.1)
    elif len(loss) == 1:
        ax.set_ylim(0, max(np.max(loss['vae_loss'])) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    if len(loss) == 2:
        plt.plot([i for i in range(1, num_epoch + 1)], loss['d_loss'], label='Discriminator', color='red', linewidth=3)
        plt.plot([i for i in range(1, num_epoch + 1)], loss['g_loss'], label='Generator', color='mediumblue', linewidth=3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    elif len(loss) == 1:
        plt.plot([i for i in range(1, num_epoch + 1)], loss['vae_loss'], label='vae_loss', color='red', linewidht=3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'vae_loss_epoch_{}.png'.format(num_epoch)))
 
    plt.close()

def create_gif(epoches, save_dir, name):
    if name == "dcgan":
        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(save_dir, 'DCGAN_result.gif'), images, fps=5)
    
        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(save_dir, 'DCGAN_result_loss.gif'), images, fps=5)

    elif name =="anomaly":
        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(save_dir, 'anoGAN_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(save_dir, 'anoGAN_result.gif'), images, fps=5)

        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(save_dir, 'anoGAN_result_loss.gif'), images, fps=5)

