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

def conditional_latent_generator(distribution, class_num, batch):
	class_labels = torch.randint(0, class_num, (batch,), dtype=torch.long)
	fake_z = distribution[class_labels[0].item()].sample((1,))
	for c in class_labels[1:]:
		fake_z = torch.cat((fake_z, distribution[c.item()].sample((1,))), dim=0)
	return fake_z, class_labels
	

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

def normalize(image):
        im = Image.open(image)
        im_normalized = im.resize((128,128), resample=Image.BILINEAR)
        #im_normalized.save(norm_path)
        return im_normalized

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(root):
        mal_class = [x for x in os.listdir(root) if x[0] != "."]

        print("Found data classes !!!")
        print(mal_class)

        images = []
        for parent, dirs, files in sorted(os.walk(root)):
                for fname in sorted(files):
                        if is_image_file(fname):
                                path = os.path.join(parent, fname)
                                x = mal_class.index(parent.split('/')[-1])
                                label = torch.LongTensor(1).fill_(x)
                                item = (path, label, path)
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
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, label, path

    def __len__(self):
        return len(self.imgs)


class CustomData(data.Dataset):
        def __init__(self, root, transform=None, loader=default_loader):
                images = make_dataset(root)
                if len(images) == 0:
                        Raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

                print("Found {} images in subfolders of: {}".format(len(images), root))

                self.root = root
                self.images = images
                self.transform = transform
                self.loader = loader

        def __getitem__(self, index):
                image, label, path = self.images[index]
                image = normalize(image)
                if self.transform is not None:
                        image = self.transform(image)

                return image, label, path

        def __len__(self):
                return len(self.images)

#Clear
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
        elif 'malware' in data_name:
                return torch.utils.data.DataLoader(CustomData(train_dir, transform), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=False)

        elif 'test' in data_name:
                return torch.utils.data.DataLoader(ImageFolder(train_dir, transform), batch_size=1, shuffle=False, num_workers=config.workers, pin_memory=False)
        else:
                return
#Clear
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

#Clear
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

    elif name =="gan":
        plt.savefig(os.path.join(save_dir, 'gan_epoch_{}.png'.format(num_epoch)))
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
    
    elif name =="gan":
        plt.savefig(os.path.join(save_dir, 'gan_epoch_{}.png'.format(num_epoch)))
        plt.close()

#Clear    
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
        plt.savefig(os.path.join(os.path.join(save_dir, "loss"), 'gan_loss_epoch_{}.png'.format(num_epoch)))
    elif len(loss) == 1:
        plt.plot([i for i in range(1, num_epoch + 1)], loss['vae_loss'], label='vae_loss', color='red', linewidht=3)
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_dir, "loss"), 'vae_loss_epoch_{}.png'.format(num_epoch)))
 
    plt.close()

#Clear
def plot_accuracy(num_epoch, epoches, save_dir, real_acc, fake_acc):
	fig, ax = plt.subplots()
	ax.set_xlim(0,epoches + 1)
	ax.set_ylim(0, max(np.max(real_pred), np.max(fake_pred)) * 1.1)
	plt.xlabel('Epoch {}'.format(num_epoch))
	plt.ylabel('Accuracy')

	plt.plot([i for i in range(1, num_epoch + 1)], real_acc, label='real data', color='red', linewidth=3)
	plt.plot([i for i in range(1, num_epoch + 1)], fake_acc, label='fake data', color='mediumblue', linewidth=3)
	plt.legend()
	plt.savefig(os.path.join(os.path.join(save_dir, "accuracy"), 'gan_accuracy_epoch_{}.png'.format(num_epoch)))

	plt.close()

#Clear
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
	

    elif name =="gan":
        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(os.path.join(save_dir,'images'), 'gan_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(os.path.join(save_dir,'images'), 'gan_images_result.gif'), images, fps=5)

        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(os.path.join(save_dir,'loss'), 'gan_loss_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(os.path.join(save_dir,'loss'), 'gan_losses_result.gif'), images, fps=5)

        images = []
        for i in range(1, epoches + 1):
            images.append(imageio.imread(os.path.join(os.path.join(save_dir,'accuracy'), 'gan_accuracy_epoch_{}.png'.format(i))))
        imageio.mimsave(os.path.join(os.path.join(save_dir,'accuracy'), 'gan_accuracy_result.gif'), images, fps=5)
	
	

