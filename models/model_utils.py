from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import torch

import torch.onnx

from collections import Iterable


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append("%s/../" % file_path)
from helper_functions.torchsummary import summary

NUM_CLASSES = 4

def _create_inputs(input_or_shape):
	if isinstance(input_or_shape, torch.Tensor):
		inputs = input_or_shape.cpu()
		inputs.requires_grad_()
	elif isinstance(input_or_shape, Iterable):
		inputs = torch.randn(*input_or_shape)
	else:
		raise ValueError(
			"Cannot recognize the argument type " + str(type(input_or_shape))
		)
	return inputs

def load_display_model(args):
	'''
	module to load and display pytorch model network
	:param args loaded through the config .ini files
	# see for example ./Ini_Files
	'''
	if args.network == 'segnet_small':
		from models.SegNet import SegNet_Small
		model = SegNet_Small(
			args.channels,
			args.classes,
			args.skip_type,
			BR_bool = args.BR,
			separable_conv = args.SC
		)
		if args.CUDA:
			model = model.cuda(args.gpu)

	summary(
		model,
		(args.channels, args.image_size, args.image_size),
		args
	)

	try:
		if args.checkpoint:
			model.load_state_dict(torch.load(args.checkpoint))
	except Exception as e:
		print (e)
		sys.exit(0)

	return model

def load_models(mode, device, args):
	"""

	:param mode: "SS" or "Discriminator"
	:param args:
	:return:
	"""

	if mode == "SS":
		if args.network == "segnet_small":
			from models.SegNet import SegNet_Small
			model = SegNet_Small(
				args.channels,
				args.classes,
				args.skip_type,
				BR_bool=args.BR,
				separable_conv=args.SC
			)
			model = model.to(device)

		summary(
			model,
			(args.channels, args.image_size, args.image_size),
			args
		)

		try:
			if args.checkpoint:
				model.load_state_dict(torch.load(args.checkpoint))

		except Exception as e:
			print(e)
			sys.exit(0)
	elif mode == "Discriminator":
		from models.discriminator import FCDiscriminator_Scalar, \
			FCDiscriminator_Scalar_BN, \
			FCDiscriminator_Pixel

		if args.discriminator == "image_level":
			model = FCDiscriminator_Scalar(num_classes=NUM_CLASSES)
		elif args.discriminator == "image_level_BN":
			model = FCDiscriminator_Scalar_BN(num_classes=NUM_CLASSES)
		elif args.discriminator == "pixel_level":
			model = FCDiscriminator_Pixel(num_classes=NUM_CLASSES, input_size = [184,184])
		else:
			raise ValueError("Invalid Discriminator mode! {}".
							 format(args.discriminator))
	else:
		raise ValueError("Invalid mode {}!".format(mode))

	return model

def lr_poly(base_lr, iter, max_iter, power):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_SS(optimizer, learning_rate, i_iter, max_steps, power):
	lr = lr_poly(learning_rate, i_iter, max_steps, power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1:
		optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_steps, power):
	lr = lr_poly(learning_rate, i_iter, max_steps, power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1:
		optimizer.param_groups[1]['lr'] = lr * 10