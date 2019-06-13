from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import os
import sys
import pickle

import numpy as np
from tensorboardX import SummaryWriter
from torchviz import make_dot

import utils.photometric_transforms as ph_transforms
from data_provider.eyeDataset import dataloader_dual
from helper_functions.analysis import generate_result_images
from helper_functions.config import ReadConfig_GAN, augment_args_GAN
from helper_functions.trainer import run_training, run_testing
from models.model_utils import load_models

from models.deeplab_multi import DeeplabMulti
from models.discriminator import FCDiscriminator

import torch
import torch.optim as optim
import torch.nn as nn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
	parser = optparse.OptionParser()
	parser.add_option("--data-root", type="str",
					  dest="data_root",
					  help="list of Path to data folder",)
	parser.add_option("--output-dir", type="str",
					  dest="output_dir",
					  default=None,
					  help="Path to folder where output images are saved")
	parser.add_option("--ini-file",type="str",
					  dest="ini_file",
					  default=None,
					  help="configuration initialization file; ")
	parser.add_option("--image-size", type="int",
					  dest="image_size",
					  default=184,
					  help="image_size scalar (currently support square images)")
	parser.add_option("--lambda-seg-source", type="float",
					  dest="lambda_seg_source",
					  default=1.0,
					  help="coefficient for CE loss w.r.t. source")
	parser.add_option("--lambda-seg-target", type="float",
					  dest="lambda_seg_target",
					  default=0.1,
					  help="coefficient for CE loss w.r.t. target")
	parser.add_option("--lambda-adv-target", type="float",
					  dest="lambda_adv_target",
					  default=0.2,
					  help="coefficient for adversarial loss")
	parser.add_option("--checkpoint",type="str",
					  dest="checkpoint",
					  default=None,
					  help="Path to pretrained model SS net")
	parser.add_option("--train",action="store_true",
					  dest="train",
					  default=False,
					  help="run training")
	parser.add_option("--val",action="store_true",
					  dest="val",
					  default=False,
					  help="run validation",)
	parser.add_option("--test",action="store_true",
					  dest="test",
					  default=False,
					  help="run testing (generate result images)",)
	parser.add_option("--num-steps-stop", dest="num_steps_stop",
					  type=int,
					  default=150000,
					  help="Number of training steps for early stopping.")
	parser.add_option("--snapshot-dir", type=str,
					  dest="snapshot_dir",
					  default="./snapshots/",
					  help="Where to save snapshots of the model.")
	parser.add_option("--log-dir", type=str,
					  dest="log_dir",
					  default="./log/image_level",
					  help="Path to the directory of log.")
	parser.add_option("--weight-decay", type=float,
					  dest="weight_decay",
					  default=0.0005,
					  help="Regularisation parameter for L2-loss.")
	parser.add_option("--iter-size", type=int,
					  dest="iter_size",
					  default=1,
					  help="Accumulate gradients for ITER_SIZE iterations.")
	parser.add_option("--train-target-SS", action="store_true",
					  dest="train_target_SS",
					  default=False,
					  help="run testing (generate result images)", )
	parser.add_option("--tensorboard", action="store_true",
					  dest="tensorboard",
					  default=False)
	parser.add_option("--log-filename", type=str,
					  dest="log_filename",
					  default="default")

	(args, opts) = parser.parse_args()

	if args.ini_file is None:
		print ('Model config file required, quitting...')
		sys.exit(0)

	if args.tensorboard:
		if not os.path.exists(args.log_dir):
			os.makedirs(args.log_dir)

		writer = SummaryWriter(args.log_dir, comment=args.log_filename)
	else:
		writer = None

	GP, TP = ReadConfig_GAN(args.ini_file)
	args = augment_args_GAN(GP, TP, args)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device
	args.transforms = [
		ph_transforms.ChangeBrightness(args.brightness_scale),
		ph_transforms.ToTensor(),
	]

	model_SS = load_models(
		mode = "SS",
		device = device,
		args = args,
	)
	# model_D1 = load_models(
	# 	mode = "Discriminator",
	# 	device = device,
	# 	args = args,
	# )
	model_D = load_models(
		mode="Discriminator",
		device=device,
		args=args,
	)

	dummy_input = torch.zeros(1, 1, 184, 184, dtype=torch.float, requires_grad=False).to(args.device)
	seg_out, _ = model_SS(dummy_input)
	make_dot(seg_out)

	# model_SS = DeeplabMulti(num_classes=args.classes)
	# model_D1 = FCDiscriminator(num_classes=args.classes)
	# model_D2 = FCDiscriminator(num_classes=args.classes)
	# interp = nn.Upsample(
	# 	size=(args.image_size, args.image_size),
	# 	mode='bilinear',
	# 	align_corners=True
	# )
	# interp_target = nn.Upsample(
	# 	size=(args.image_size, args.image_size),
	# 	mode='bilinear',
	# 	align_corners=True
	# )

	### use SGD for source only SS ###
	# optimizer_SS = optim.SGD(
	# 	model_SS.parameters(),
	# 	lr = args.lr_SS,
	# 	momentum = args.momentum,
	# 	weight_decay = args.weight_decay,
	# )
	# optimizer_SS.zero_grad()
	### use SGD for source only SS ###

	optimizer_SS = optim.Adam(
		model_SS.parameters(),
		lr=args.lr_SS,
		betas=(0.9, 0.999),
	)
	optimizer_SS.zero_grad()

	optimizer_D = optim.Adam(
		model_D.parameters(),
		lr=args.lr_D,
		betas=(0.9, 0.999),
	)
	optimizer_D.zero_grad()

	print("Input Arguments: {}".format(args))

	if args.train:
		trainset_openeds, trainset_calipso, trainloader_openeds, trainloader_calipso = dataloader_dual(
			args,
			type="train",
		)
		class_weight_openeds = 1.0 / trainset_openeds.get_class_probability().to(device)
		criterion_CE_openeds = torch.nn.CrossEntropyLoss(
			weight = class_weight_openeds,
		).to(device)

		class_weight_calipso = 1.0 / trainset_calipso.get_class_probability().to(device)
		criterion_CE_calipso = torch.nn.CrossEntropyLoss(
			weight = class_weight_calipso,
		).to(device)

		criterion_ADV = torch.nn.BCEWithLogitsLoss().to(device)
		criterion_D = torch.nn.BCEWithLogitsLoss().to(device)

		args.total_iterations = int(args.num_epochs *
									trainset_openeds.__len__() / args.batch_size)
		args.total_source = trainset_openeds.__len__()

	if args.val:
		valset_calipso, valloader_calipso = dataloader_dual(
			args,
			type = "val",
		)

	if args.train:

		model_SS.train()
		model_D.train()

		model_SS.to(args.device)
		model_D.to(args.device)

		bce_loss = torch.nn.BCEWithLogitsLoss()
		seg_loss_source = torch.nn.CrossEntropyLoss(weight=class_weight_openeds)
		seg_loss_target = torch.nn.CrossEntropyLoss(weight=class_weight_calipso)

		trainloader_iter = enumerate(trainloader_openeds)
		targetloader_iter = enumerate(trainloader_calipso)


		val_loss = run_training(
			trainloader_source = trainloader_openeds,
			trainloader_target = trainloader_calipso,
			trainloader_iter = trainloader_iter,
			targetloader_iter = targetloader_iter,
			val_loader = valloader_calipso,
			model_SS = model_SS,
			model_D = model_D,
			bce_loss = bce_loss,
			seg_loss_source = seg_loss_source,
			seg_loss_target = seg_loss_target,
			optimizer_SS = optimizer_SS,
			optimizer_D = optimizer_D,
			writer = writer,
			args = args,
		)

		try:
			o = open("%s/loss_val.pkl" % args.output_dir, "wb")
			pickle.dump(
				[
					val_loss
				],
				o,
				protocol=2,
			)
			o.close()
		except FileNotFoundError as e:
			print(e)

	if args.val:
		pm = run_testing(
			val_loader = valloader_calipso,
			model = model_SS,
			args = args,
		)
		print('Global Mean Accuracy:', np.array(pm.GA).mean())
		print('Mean IOU:', np.array(pm.IOU).mean())
		try:
			o = open("%s/metrics.pkl" % args.output_dir, "wb")
			pickle.dump([pm.GA,pm.CA, pm.Precision, pm.Recall, pm.F1, pm.IOU],o,protocol=2)
			o.close()
		except FileNotFoundError as e:
			print (e)

	if args.test:
		testset_calipso, testloader_calipso = dataloader_dual(
			args,
			type = "test",
		)

		pm_test, data_test, preds_test, acts_test = run_testing(
			val_loader = testloader_calipso,
			model = model_SS,
			args = args,
			get_images = True,
		)
		print('Global Mean Accuracy:', np.array(pm_test.GA).mean())
		print('Mean IOU:', np.array(pm_test.IOU).mean())
		print('Mean Recall:', np.array(pm_test.Recall).mean())
		print('Mean Precision:', np.array(pm_test.Precision).mean())
		print('Mean F1:', np.array(pm_test.F1).mean())

		try:
			if not os.path.isdir(args.output_dir):
				os.mkdir(args.output_dir)
		except:
			print('Output directory not specified')

		print("Generating predicted images ...")
		for i in range(len(data_test)):
			generate_result_images(
				input_image = data_test[i],
				target_image = acts_test[i],
				pred_image = preds_test[i],
				args=args,
				iou = pm_test.IOU[i],
				count = i,
			)

