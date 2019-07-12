import functools

import torch.nn as nn
from torch.nn import init

class Identity(nn.Module):
	def forward(self, x):
		return x

def get_norm_layer(norm_type='instance'):
	"""Return a normalization layer
	Parameters:
		norm_type (str) -- the name of the normalization layer: batch | instance | none
	For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
	For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
	"""
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'none':
		norm_layer = lambda x: Identity()
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
	"""Initialize network weights.
	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, device, init_type='normal', init_gain=0.02):
	"""Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
	Parameters:
		net (network)      -- the network to be initialized
		init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		gain (float)       -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
	Return an initialized network.
	"""
	net.to(device)
	init_weights(net, init_type, init_gain=init_gain)
	return net

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x


class PatchGAN(nn.Module):
	"""Defines a PatchGAN discriminator"""
	"""Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py """

	def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator
		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			n_layers (int)  -- the number of conv layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(PatchGAN, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func != nn.BatchNorm2d
		else:
			use_bias = norm_layer != nn.BatchNorm2d

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [
			nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		"""Standard forward."""
		return self.model(input)


class FCDiscriminator_Scalar(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_Scalar, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf*1, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*1, 1, kernel_size=5, stride=1, padding=0)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		x = self.conv6(x)

		return x

class FCDiscriminator_Pixel(nn.Module):

	def __init__(self, num_classes, input_size, ndf = 64):
		super(FCDiscriminator_Pixel, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
		self.interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=False)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.interp(x)

		return x

class FCDiscriminator_Scalar_BN(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_Scalar_BN, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf*1, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*1, 1, kernel_size=5, stride=1, padding=0)

		self.bn3 = nn.BatchNorm2d(ndf*4)
		self.bn4 = nn.BatchNorm2d(ndf*8)
		self.bn5 = nn.BatchNorm2d(ndf*1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.bn4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.bn5(x)
		x = self.leaky_relu(x)
		x = self.conv6(x)

		return x