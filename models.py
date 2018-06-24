import torch
import torch.nn as nn

# Generator
class Decoder(nn.Module):
	def __init__(self, z_dim, c_dim, gf_dim):
		super(Decoder, self).__init__()

		self.convTrans0 = nn.ConvTranspose2d(z_dim, gf_dim*8, 4, 1, 0, bias=False)
		self.bn0 = nn.BatchNorm2d(gf_dim*8)
		self.relu0 = nn.ReLU(inplace=True)
		
		self.convTrans1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(gf_dim*4)
		self.relu1 = nn.ReLU(inplace=True)

		self.convTrans2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(gf_dim*2)
		self.relu2 = nn.ReLU(inplace=True)

		self.convTrans3 = nn.ConvTranspose2d(gf_dim*2, gf_dim, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(gf_dim)
		self.relu3 = nn.ReLU(inplace=True)

		self.convTrans4 = nn.ConvTranspose2d(gf_dim, c_dim, 4, 2, 1, bias=False)
		self.tanh = nn.Tanh()

		for m in self.modules():
				if isinstance(m, nn.ConvTranspose2d):
						m.weight.data.normal_(0.0, 0.02)
						if m.bias is not None:
								m.bias.data.zero_()

	def forward(self, z):
		h0 = self.relu0(self.bn0(self.convTrans0(z)))
		h1 = self.relu1(self.bn1(self.convTrans1(h0)))
		h2 = self.relu2(self.bn2(self.convTrans2(h1)))
		h3 = self.relu3(self.bn3(self.convTrans3(h2)))
		h4 = self.convTrans4(h3)
		output = self.tanh(h4)
		return output # (c_dim, 64, 64)

# Discriminator
class Encoder(nn.Module): 
	def __init__(self, z_dim, c_dim, df_dim):
		super(Encoder, self).__init__()
		self.df_dim = df_dim

		self.conv0 = nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False)
		self.relu0 = nn.LeakyReLU(0.2, inplace=True)
		
		self.conv1 = nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(df_dim*2)
		self.relu1 = nn.LeakyReLU(0.2, inplace=True)

		self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(df_dim*4)
		self.relu2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(df_dim*8)
		self.relu3 = nn.LeakyReLU(0.2, inplace=True)

		self.fc_z1 = nn.Linear(df_dim*8*4*4, z_dim)
		self.fc_z2 = nn.Linear(df_dim*8*4*4, z_dim)

		#self.conv4 = nn.Conv2d(df_dim*8, 1, 4, 1, 0, bias=False)

		for m in self.modules():
				if isinstance(m, nn.Conv2d):
						m.weight.data.normal_(0.0, 0.02)
						if m.bias is not None:
								m.bias.data.zero_()

	def forward(self, input):
		h0 = self.relu0(self.conv0(input))
		h1 = self.relu1(self.bn1(self.conv1(h0)))
		h2 = self.relu2(self.bn2(self.conv2(h1)))
		h3 = self.relu3(self.bn3(self.conv3(h2)))
		
		mu = self.fc_z1(h3.view(-1, self.df_dim*8*4*4))	# (1, 128*8*4*4)
		sigma = self.fc_z2(h3.view(-1, self.df_dim*8*4*4))
		return mu,sigma # by squeeze, get just float not float Tenosor


class Generator(nn.Module):
        def __init__(self, z_dim, c_dim, gf_dim):
                super(Generator, self).__init__()

                self.convTrans0 = nn.ConvTranspose2d(z_dim, gf_dim*8, 4, 1, 0, bias=False)
                self.bn0 = nn.BatchNorm2d(gf_dim*8)
                self.relu0 = nn.ReLU(inplace=True)

                self.convTrans1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(gf_dim*4)
                self.relu1 = nn.ReLU(inplace=True)

                self.convTrans2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 4, 2, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(gf_dim*2)
                self.relu2 = nn.ReLU(inplace=True)

                self.convTrans3 = nn.ConvTranspose2d(gf_dim*2, gf_dim, 4, 2, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(gf_dim)
                self.relu3 = nn.ReLU(inplace=True)

                self.convTrans4 = nn.ConvTranspose2d(gf_dim, c_dim, 4, 2, 1, bias=False)
                self.tanh = nn.Tanh()


        def forward(self, z):
                h0 = self.relu0(self.bn0(self.convTrans0(z)))
                h1 = self.relu1(self.bn1(self.convTrans1(h0)))
                h2 = self.relu2(self.bn2(self.convTrans2(h1)))
                h3 = self.relu3(self.bn3(self.convTrans3(h2)))
                h4 = self.convTrans4(h3)
                output = self.tanh(h4)
                return output # (c_dim, 64, 64)

class _ganLogits(nn.Module):
    '''
    Layer of the GAN logits of the discriminator
    The layer gets class logits as inputs and calculates GAN logits to
    differentiate real and fake images in a numerical stable way
    '''
    def __init__(self, num_classes):
        '''
        :param num_classes: Number of real data classes (10 for SVHN)
        '''
        super(_ganLogits, self).__init__()
        self.num_classes = num_classes

    def forward(self, class_logits):
        '''
        :param class_logits: Unscaled log probabilities of house numbers
        '''

        # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).
        # Keep in mind that class_logits gives you the probability distribution over all the real
        # classes and the fake class. You need to work out how to transform this multiclass softmax
        # distribution into a binary real-vs-fake decision that can be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)
        real_class_logits, fake_class_logits = torch.split(class_logits, self.num_classes, dim=1)
        fake_class_logits = torch.squeeze(fake_class_logits)

        max_val, _ = torch.max(real_class_logits, 1, keepdim=True)
        stable_class_logits = real_class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.log(torch.sum(torch.exp(stable_class_logits), 1)) + max_val - fake_class_logits

        return gan_logits	# [128]

class Discriminator(nn.Module):
        def __init__(self, z_dim, c_dim, df_dim, class_num):
                super(Discriminator, self).__init__()
                self.df_dim = df_dim

                self.conv0 = nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False)
                self.relu0 = nn.LeakyReLU(0.2, inplace=True)

                self.conv1 = nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(df_dim*2)
                self.relu1 = nn.LeakyReLU(0.2, inplace=True)

                self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(df_dim*4)
                self.relu2 = nn.LeakyReLU(0.2, inplace=True)

                self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 4, 2, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(df_dim*8)
                self.relu3 = nn.LeakyReLU(0.2, inplace=True)

		#self.fc_z = nn.Linear(df_dim*8*4*4, z_dim)
                self.fc_aux = nn.Linear(df_dim*8*4*4, class_num+1)
                self.softmax = nn.LogSoftmax()

                for m in self.modules():
                        if isinstance(m, nn.Linear):
                                m.weight.data.normal_(0.0, 0.02)
                                if m.bias is not None:
                                        m.bias.data.zero_()


        def forward(self, input):
                h0 = self.relu0(self.conv0(input))
                h1 = self.relu1(self.bn1(self.conv1(h0)))
                h2 = self.relu2(self.bn2(self.conv2(h1)))
                h3 = self.relu3(self.bn3(self.conv3(h2)))
                #cl = self.class_logistics(h3.view(-1, self.df_dim*8*4*4))
                #gl = self.gan_logistics(cl)    
                output = self.softmax(self.fc_aux(h3.view(-1, self.df_dim*8*4*4)))
                return h3, output
                #return h3, output.view(-1,1).squeeze(1) # by squeeze, get just float not float Tenosor
