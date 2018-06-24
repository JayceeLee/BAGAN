import os
import scipy.misc
import time
import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as mn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch import optim
from torch.autograd import Variable
from utils import *
from models import *

if __name__ == "__main__":
	use_cuda = torch.cuda.is_available()
	gpu = 0

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, dest='config', help='the name of yaml file to set parameter', default='config.yaml')
	parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model", action='store_true', default=False)
	parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true', default=True)
	parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
	parser.add_argument('--train_dir', type=str, dest='train_dir', help='the path of train data')
	parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir', default='checkpoint')
	parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir', default='sample')
	parser.add_argument('--distribution_dir', type=str, dest='distribution_dir', help='the path of class distribution dir', default='distribution')
	parser.add_argument('--test_dir', type=str, dest='test_dir', help='the path of anomaly test data')
	parser.add_argument('--test_result_dir', type=str, dest='test_result_dir', help='the path of anomaly test result dir')

	args = parser.parse_args()
	config = Config(args.config)

	if not os.path.exists(args.save_dir):
		os.mkdir(os.path.join(args.root_dir, args.save_dir))
	transform = transforms.Compose([
        transforms.CenterCrop(150),
        transforms.Scale(config.image_size),
        transforms.ToTensor(),                     
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	if config.dataset == 'mnist':
		train_loader = load_data(os.path.join(args.root_dir, args.train_dir), transform, 'mnist', config)
	elif config.dataset == 'celebA':
		train_loader = load_data(os.path.join('/workspace/anoGAN', args.train_dir), transform, 'celebA', config)
	elif config.dataset == 'cifar10':
		train_loader = load_data(os.path.join(args.root_dir, args.train_dir), transform, 'cifar10', config)

	decoder = Decoder(config.z_dim, config.c_dim, config.gf_dim)
	encoder = Encoder(config.z_dim, config.c_dim, config.df_dim)

	if not args.pretrained:
		if use_cuda:
			decoder = decoder.cuda(gpu)
			encoder = encoder.cuda(gpu)

		# WHY BECLoss() - only need to determine fake/real for Discriminator
		criterion = nn.BCELoss()
		if use_cuda:
			criterion = criterion.cuda(gpu)

		optimizerE = torch.optim.Adam(encoder.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
		optimizerD = torch.optim.Adam(decoder.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
	
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()

		fixed_noise = torch.FloatTensor(8 * 8, config.z_dim, 1, 1).normal_(0, 1)
		if use_cuda:
			fixed_noise = fixed_noise.cuda(gpu)
		with torch.no_grad():
			fixed_noisev = fixed_noise

		end = time.time()
		
		encoder.train()
		decoder.train()
		loss_list = []

		
		criterion = nn.MSELoss(size_average=False)	
		for epoch in range(config.epoches):
			for i, (input, label) in enumerate(train_loader):
				#l Update 'D' : max log(D(x)) + log(1-D(G(z)))
				data_time.update(time.time()-end)
			
				batch_size = input.size(0)
				if use_cuda:
					input = input.cuda(gpu)
				
				mu, log_sigmoid = encoder(input)
				# reparameterization
				std = torch.exp(log_sigmoid/2)
				eps = torch.randn_like(std)
				z = mu + eps * std
				z = z.view(-1, config.z_dim, 1, 1)	
				if use_cuda:
					z = z.cuda(gpu)

				# reconstruct image
				x_reconstruct = decoder(z)

				# reconstruct_loss + KL_divergence
				reconstruct_loss = criterion(x_reconstruct, input)
				kl_div = -0.5 * torch.sum(1+log_sigmoid-mu.pow(2)-log_sigmoid.exp())
				loss = reconstruct_loss + kl_div
				losses.update(loss.item())	
				optimizerE.zero_grad()
				optimizerD.zero_grad()
				loss.backward()
				optimizerE.step()
				optimizerD.step()

				batch_time.update(time.time()-end)
				end = time.time()
		
				# log every 100th train data of train_loader - display(100)	
				if (i+1) % config.display == 0:
					print_vae_log(epoch+1, config.epoches, i+1, len(train_loader), config.base_lr, config.display, batch_time, data_time, losses)
					# Is it Continous ???
					batch_time.reset()
					data_time.reset()
				# log every 1 epoch (all of train_loader)
				elif (i+1) == len(train_loader):
					print_vae_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr, (i + 1) % config.display, batch_time, data_time, losses)
					batch_time.reset()
					data_time.reset()

			# log every 1 epoch
			loss_list.append(losses.avg)
			losses.reset()

			plot_result(decoder, fixed_noisev, config.image_size, epoch + 1, args.save_dir, 'vae', is_gray=(config.c_dim == 1))
			#plot_loss(epoch+1, config.epoches, args.save_dir, vae_loss=loss_list)
			# save the D and G.
			save_checkpoint({'epoch': epoch, 'state_dict': encoder.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,"vae"), 'encoder_epoch_{}'.format(epoch)))
			save_checkpoint({'epoch': epoch, 'state_dict': decoder.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,"vae"), 'decoder_epoch_{}'.format(epoch)))
	
		create_gif(config.epoches, args.save_dir, 'vae')

	## Class Conditional Generator - Pretrained Model"
	else:
		print("Class Conditional Generator - Use Pretrained Model")
		if use_cuda:
			encoder = encoder.cuda(gpu)
			decoder = decoder.cuda(gpu)
		encoder.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "vae"), "encoder_epoch_"+ str(config.epoches-1) + ".pth.tar"))['state_dict'])
		decoder.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "vae"), "decoder_epoch_"+ str(config.epoches-1) + ".pth.tar"))['state_dict'])
		#Z = np.empty([config.class_num, config.z_dim], dtype=float)
		# Z : [label-1, labe-2, ... ]
		# Z[label-1] : [[z1], [z2], ... ] (#labeld_data, #z_dim)
		encoder.eval()
		decoder.eval()
		Z = []
		with torch.no_grad():
			for i in range(config.class_num):
				Z.append(torch.zeros((1, config.z_dim), dtype=torch.float)) # Z : [class_num, z_dim]
		
			for i, (input, label) in enumerate(train_loader):
				if use_cuda:
					input = input.cuda(gpu)
				mu, log_sigmoid = encoder(input)
				std = torch.exp(log_sigmoid/2)
				eps = torch.randn_like(std)
				z = mu + eps * std
				z = z.view(-1, 1, config.z_dim)
				Z = batch2one(Z, label, z, config.class_num)

			N = []
			for i in range(config.class_num):
				label_mean = torch.mean(Z[i][1:], dim=0)
				label_cov = torch.from_numpy(np.cov(Z[i][1:].numpy(), rowvar=False))
				#print("{}th Z : {}".format(i+1, Z[i][1:].shape))
				#print("{}-class  mean : {}".format(i+1, label_mean.shape))
				#print("{}-class covariance : {}".format(i+1, label_cov.shape))
				m = mn.MultivariateNormal(label_mean, label_cov)
				sample = m.sample((64,))
				print(sample.shape)
				if use_cuda:
					sample = sample.cuda(gpu)
				fake = decoder(sample.view(-1, config.z_dim, 1, 1))
				plot_result2(fake, config.image_size, i, 'data/gan/samples', 'ssgan', is_gray=(config.c_dim == 1))
				N.append(m)

			torch.save({'distribution': N}, os.path.join(args.distribution_dir, 'class_distribution')+'.dt')
