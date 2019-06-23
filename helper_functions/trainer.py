from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import sys
import time

import numpy as np
import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F


from .torchsummary import summary
from .metrics import evaluate_segmentation
from .trainer_utils import adjust_learning_rate, Performance_Metrics, make_D_label

import matplotlib.pyplot as plt
plt.switch_backend('agg')

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
sys.path.append("%s/../../.." % file_path)
import utils.photometric_transforms as ph_transforms
from models.SegNet import SegNet_Small

INPUT_CHANNELS = 3
NUM_CLASSES = 4


def validate(
		val_loader,
		model,
		epoch,
		args,
		val_loss = float("inf")
):
	'''
	Pytorch model validation module
	:param val_loader: pytorch dataloader
	:param model: pytorch model
	:param criterion: loss function
	:param args: input arguments from __main__
	:param epoch: epoch counter used to display training progress
	:param val_loss: logs loss prior to invocation of train on batch
	'''
	seg_loss = torch.nn.CrossEntropyLoss().to(args.device)
	model.eval()
	t_start = time.time()

	loss_f = 0
	save_img_bool = True
	batch_idx_img = np.random.randint(0,val_loader.__len__())
	for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total=val_loader.__len__()):
		image, label = data
		image, label = Variable(image).float(), \
					  Variable(label).type(torch.LongTensor)

		image, label = image.to(args.device), label.to(args.device)


		with torch.set_grad_enabled(False):
			predicted_tensor, softmaxed_tensor = model(image)

		loss = seg_loss(predicted_tensor, label)
		loss_f += loss.item()

		delta = time.time() - t_start

		try:
			if not os.path.isdir(args.output_dir):
				os.mkdir(args.output_dir)
		except TypeError as e:
			print (e)
			save_img_bool=False

		id_img = np.random.randint(0, args.batch_size)
		for idx, predicted_mask in enumerate(predicted_tensor):
			if idx == id_img and batch_idx == batch_idx_img:
				input_image, target_mask = image[idx], label[idx]
				c,h,w = input_image.size()
				if save_img_bool:
					fig = plt.figure()
					a = fig.add_subplot(1,3,1)
					if c == 1:
						plt.imshow(input_image.detach().cpu().transpose(1,2).transpose(0, 2)[:,:,0],cmap='gray')
					else:
						plt.imshow(input_image.detach().cpu().transpose(1,2).transpose(0, 2),cmap='gray')
					a.set_title('Input Image')

					a = fig.add_subplot(1,3,2)
					predicted_mx = predicted_mask.detach().cpu().numpy()
					predicted_mx = predicted_mx.argmax(axis=0)
					plt.imshow(predicted_mx)
					a.set_title('Predicted Mask')

					a = fig.add_subplot(1,3,3)
					target_mx = target_mask.detach().cpu().numpy()
					plt.imshow(target_mx)
					a.set_title('Ground Truth')
					fig.savefig(os.path.join(args.output_dir, "prediction_{}_{}_{}.png".format(epoch+1, batch_idx, idx)))
					plt.close(fig)

	print("Epoch #{}\t Val Loss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f* 1.0 / val_loader.__len__(), delta))

	new_val_loss=loss_f * 1.0 / val_loader.__len__()


	if new_val_loss<val_loss:
		print(val_loss, ',', new_val_loss)
		print('saving checkpoint ....')
		if not os.path.isdir(args.snapshot_dir):
			os.mkdir(args.snapshot_dir)
		torch.save(model.state_dict(), os.path.join(args.snapshot_dir, "model_val_best.pth"))

	return new_val_loss


# def run_training(
# 		trainloader_source,
# 		trainloader_target,
# 		trainloader_iter,
# 		targetloader_iter,
# 		val_loader,
# 		model_SS,
# 		model_D,
# 		bce_loss,
# 		seg_loss_source,
# 		seg_loss_target,
# 		optimizer_SS,
# 		optimizer_D,
# 		writer,
# 		args,
# ):
# 	# labels for adversarial training
# 	source_label = 0
# 	target_label = 1
#
# 	val_loss = []
# 	val_loss_f = float("inf")
# 	loss_seg_min = float("inf")
#
# 	# set up tensor board
# 	if args.tensorboard:
# 		if not os.path.exists(args.log_dir):
# 			os.makedirs(args.log_dir)
#
# 		# dummy_input = Variable(torch.Tensor(1, 1, 184, 184).to(args.device), requires_grad=False)
# 		# writer.add_graph(model_SS, dummy_input)
#
# 	for i_iter in range(args.total_iterations):
#
# 		loss_seg_source_value = 0
# 		loss_seg_target_value = 0
# 		loss_adv_target_value = 0
# 		loss_D_value = 0
#
# 		optimizer_SS.zero_grad()
# 		optimizer_D.zero_grad()
#
# 		adjust_learning_rate(
# 			optimizer= optimizer_SS,
# 			learning_rate = args.lr_SS,
# 			i_iter = i_iter,
# 			max_steps = args.total_iterations,
# 			power = 0.9
# 		)
#
# 		adjust_learning_rate(
# 			optimizer=optimizer_D,
# 			learning_rate=args.lr_D,
# 			i_iter=i_iter,
# 			max_steps=args.total_iterations,
# 			power=0.9
# 		)
#
# 		for sub_i in range(args.iter_size):
#
# 			# train G
#
# 			for param in model_D.parameters():
# 				param.requires_grad = False
#
# 			# train with source
#
# 			try:
# 				_, batch = next(trainloader_iter)
# 			except StopIteration:
# 				trainloader_iter = enumerate(trainloader_source)
# 				_, batch = next(trainloader_iter)
#
# 			images, labels = batch
# 			images = images.to(args.device)
# 			labels = labels.long().to(args.device)
#
# 			pred, pred_softmax = model_SS(images)
# 			loss_seg_source = seg_loss_source(pred, labels)
#
# 			loss_seg_source_value += loss_seg_source.item() / args.iter_size
#
# 			# train with target
#
# 			try:
# 				_, batch = next(targetloader_iter)
# 			except StopIteration:
# 				targetloader_iter = enumerate(trainloader_target)
# 				_, batch = next(targetloader_iter)
#
# 			images, labels = batch
# 			images = images.to(args.device)
# 			labels = labels.long().to(args.device)
#
# 			pred_target, pred_softmax_target = model_SS(images)
#
# 			loss_seg_target = seg_loss_target(pred_target, labels)
# 			loss_seg_target_value += loss_seg_target.item() / args.iter_size
#
# 			if args.train_target_SS:
# 				current_loss_seg = loss_seg_source.item() + loss_seg_target.item()
# 			else:
# 				current_loss_seg = loss_seg_source.item()
#
# 			D_out = model_D(pred_softmax_target)
#
# 			generated_label = make_D_label(
# 				input = D_out,
# 				value = source_label,
# 				device = args.device,
# 				random = False,
# 			)
#
# 			loss_adv_target = bce_loss(D_out, generated_label)
#
# 			loss = args.lambda_seg_source * loss_seg_source + \
# 				   args.lambda_seg_target * loss_seg_target + \
# 				   args.lambda_adv_target * loss_adv_target
# 			loss = loss / args.iter_size
# 			loss.backward()
# 			loss_adv_target_value += loss_adv_target.item() / args.iter_size
#
# 			# train D
#
# 			for param in model_D.parameters():
# 				param.requires_grad = True
#
# 			# train with source
#
# 			pred_softmax = pred_softmax.detach()
# 			D_out = model_D(pred_softmax)
#
# 			generated_label = make_D_label(
# 				input=D_out,
# 				value=source_label,
# 				device=args.device,
# 				random=True,
# 			)
# 			loss_D = bce_loss(D_out, generated_label)
# 			loss_D = loss_D / args.iter_size / 2
# 			loss_D.backward()
#
# 			loss_D_value += loss_D.item()
#
# 			# train with target
#
# 			pred_softmax_target = pred_softmax_target.detach()
# 			D_out = model_D(pred_softmax_target)
#
# 			generated_label = make_D_label(
# 				input = D_out,
# 				value = target_label,
# 				device = args.device,
# 				random = True,
# 			)
#
# 			loss_D = bce_loss(D_out, generated_label)
# 			loss_D = loss_D / args.iter_size / 2
# 			loss_D.backward()
#
# 			loss_D_value += loss_D.item()
#
# 		optimizer_SS.step()
# 		optimizer_D.step()
#
# 		if args.tensorboard:
# 			scalar_info = {
# 				'loss_seg_source': loss_seg_source_value,
# 				'loss_seg_target': loss_seg_target_value,
# 				'loss_adv_target': loss_adv_target_value,
# 				'loss_D': loss_D_value,
# 			}
#
# 			for key, val in scalar_info.items():
# 				writer.add_scalar(key, val, i_iter)
#
# 		print('iter = {0:8d}/{1:8d} '
# 			  'loss_seg_source = {2:.3f} loss_seg_target = {3:.3f} '
# 			  'loss_adv2 = {4:.3f} '
# 			  'loss_D = {5:.3f}'.format(i_iter, args.total_iterations,
# 										loss_seg_source_value, loss_seg_target_value,
# 										loss_adv_target_value,
# 										loss_D_value))
#
# 		every_5_epoch = int((5 * args.total_source) / args.batch_size)
# 		current_epoch = int(i_iter * args.batch_size / args.total_source)
# 		if i_iter % every_5_epoch == 0:
# 			# evaluate on validation
# 			val_loss_f = validate(
# 				val_loader=val_loader,
# 				model=model_SS,
# 				epoch=current_epoch,
# 				args=args,
# 				val_loss=val_loss_f,
# 			)
# 			val_loss.append(val_loss_f)
# 			val_loss_f = np.min(np.array(val_loss))
# 			if args.tensorboard:
# 				scalar_info = {
# 					'loss_val': val_loss_f,
# 				}
# 				for key, val in scalar_info.items():
# 					writer.add_scalar(key, val, i_iter)
#
# 			# determine optimal model and save it
# 			is_better_ss = current_loss_seg < loss_seg_min
# 			if is_better_ss:
# 				loss_seg_min = current_loss_seg
# 				if not os.path.isdir(args.snapshot_dir):
# 					os.mkdir(args.snapshot_dir)
# 				torch.save(model_SS.state_dict(), os.path.join(args.snapshot_dir, "modelSS_train_best.pth"))
#
#
# 	current_epoch = int(args.total_iterations * args.batch_size / args.total_source)
# 	val_loss_f = validate(
# 		val_loader=val_loader,
# 		model=model_SS,
# 		epoch=current_epoch,
# 		args=args,
# 		val_loss=val_loss_f,
# 	)
# 	val_loss.append(val_loss_f)
# 	if args.tensorboard:
# 		scalar_info = {
# 			'loss_val': val_loss_f,
# 		}
# 		for key, val in scalar_info.items():
# 			writer.add_scalar(key, val, args.total_iterations)
#
# 	if args.tensorboard:
# 		writer.close()
#
# 	return val_loss

def run_testing(
		val_loader,
		model,
		args,
		get_images = False,
):
	'''
	Module to run testing on trained pytorch model
	current implementation is not memory efficient..
	do not expect testing code to run on significantly larger dataset at present...
	:param val_loader: dataloader
	:param model: pretrained pytorch model
	:param: args: input arguments from __main__
	:param get_images: Boolean, True implies model generated segmentation mask results will be dumped out
	'''
	model.eval()
	# CUDA = args.gpu is not None
	# preds = []
	# acts = []
	# inp_data = []

	## Compute Metrics:
	Global_Accuracy=[]; Class_Accuracy=[]; Precision=[]; Recall=[]; F1=[]; IOU=[]
	pm = Performance_Metrics(
		Global_Accuracy,
		Class_Accuracy,
		Precision,
		Recall,
		F1,
		IOU,
	)

	for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total = val_loader.__len__()):
		image, label = data
		image, label = Variable(image).float(), \
					  Variable(label).type(torch.LongTensor)

		image, label = image.to(args.device), label.to(args.device)

		with torch.set_grad_enabled(False):
			predicted_tensor, softmaxed_tensor = model(image)

		image = image.detach().cpu().numpy()
		label = label.detach().cpu().numpy()
		pred = np.argmax(predicted_tensor.detach().cpu().numpy(), axis=1)

		if batch_idx == 0:
			collate_image = image.copy()
			collate_preds = pred.copy()
			collate_labels = label.copy()
		else:
			collate_image = np.vstack([collate_image, image.copy()])
			collate_preds = np.vstack([collate_preds, pred.copy()])
			collate_labels = np.vstack([collate_labels, label.copy()])

		for idx in np.arange(pred.shape[0]):
			ga, ca, prec, rec, f1, iou = evaluate_segmentation(
				pred[idx, :],
				label[idx, :],
				NUM_CLASSES,
			)
			pm.GA.append(ga)
			pm.CA.append(ca)
			pm.Precision.append(prec)
			pm.Recall.append(rec)
			pm.F1.append(f1)
			pm.IOU.append(iou)

	if get_images:
		return pm, collate_image, collate_preds, collate_labels
	else:
		return pm

def run_training(
		trainloader_source,
		trainloader_target,
		trainloader_iter,
		targetloader_iter,
		val_loader,
		model_SS,
		model_D,
		bce_loss,
		seg_loss_source,
		seg_loss_target,
		semi_loss_target,
		optimizer_SS,
		optimizer_D,
		writer,
		args,
):
	# labels for adversarial training
	source_label = 0
	target_label = 1

	val_loss = []
	val_loss_f = float("inf")
	loss_seg_min = float("inf")

	# set up tensor board
	if args.tensorboard:
		if not os.path.exists(args.log_dir):
			os.makedirs(args.log_dir)

	for i_iter in range(args.total_iterations):

		loss_seg_source_value = 0
		loss_seg_target_value = 0
		loss_adv_target_value = 0
		loss_semi_target_value = 0
		loss_D_value = 0

		optimizer_SS.zero_grad()
		optimizer_D.zero_grad()

		adjust_learning_rate(
			optimizer= optimizer_SS,
			learning_rate = args.lr_SS,
			i_iter = i_iter,
			max_steps = args.total_iterations,
			power = 0.9
		)

		adjust_learning_rate(
			optimizer=optimizer_D,
			learning_rate=args.lr_D,
			i_iter=i_iter,
			max_steps=args.total_iterations,
			power=0.9
		)

		for sub_i in range(args.iter_size):

			# train G

			for param in model_D.parameters():
				param.requires_grad = False

			# train with source

			try:
				_, batch = next(trainloader_iter)
			except StopIteration:
				trainloader_iter = enumerate(trainloader_source)
				_, batch = next(trainloader_iter)

			images, labels = batch
			images = images.to(args.device)
			labels = labels.long().to(args.device)

			pred, pred_softmax = model_SS(images)
			loss_seg_source = seg_loss_source(pred, labels)

			loss_seg_source_value += loss_seg_source.item() / args.iter_size

			# train with target

			try:
				_, batch = next(targetloader_iter)
			except StopIteration:
				targetloader_iter = enumerate(trainloader_target)
				_, batch = next(targetloader_iter)

			images, labels = batch
			images = images.to(args.device)
			labels = labels.long().to(args.device)

			pred_target, pred_softmax_target = model_SS(images)

			loss_seg_target = seg_loss_target(pred_target, labels)
			loss_seg_target_value += loss_seg_target.item() / args.iter_size

			if args.train_target_SS:
				current_loss_seg = loss_seg_source.item() + loss_seg_target.item()
			else:
				current_loss_seg = loss_seg_source.item()

			# semi loss for target
			if args.semi_start != 0 and i_iter >= args.semi_start:
				pred_confidence_value = pred_softmax_target.data.cpu().numpy().max(axis=1)
				semi_ignore_mask = (pred_confidence_value < args.mask_T)
				semi_gt = pred_softmax_target.data.cpu().numpy().argmax(axis=1)  # NxCxWxH
				semi_gt[semi_ignore_mask] = 255

				semi_ratio = 1.0 - float(semi_ignore_mask.sum()) / semi_ignore_mask.size
				# print('semi ratio: {:.4f}'.format(semi_ratio))
				if semi_ratio == 0.0:
					loss_semi_target_value += 0
					loss_semi = None
				else:
					semi_gt = torch.LongTensor(semi_gt).to(args.device)
					loss_semi_target = semi_loss_target(pred_target, semi_gt)
					loss_semi_target_value += loss_semi_target.item() / args.iter_size
					loss_semi = args.lambda_semi_target * loss_semi_target
			else:
				loss_semi = None

			D_out = model_D(pred_softmax_target)

			generated_label = make_D_label(
				input=D_out,
				value=source_label,
				device=args.device,
				random=False,
			)

			loss_adv_target = bce_loss(D_out, generated_label)
			loss_adv_target_value += loss_adv_target.item() / args.iter_size

			if loss_semi is None:
				loss = args.lambda_seg_source * loss_seg_source + \
					   args.lambda_seg_target * loss_seg_target + \
					   args.lambda_adv_target * loss_adv_target
			else:
				loss = args.lambda_seg_source * loss_seg_source + \
					   args.lambda_seg_target * loss_seg_target + \
					   args.lambda_adv_target * loss_adv_target + \
					   args.lambda_semi_target * loss_semi

			loss = loss / args.iter_size
			loss.backward()

			# train D

			for param in model_D.parameters():
				param.requires_grad = True

			# train with source

			pred_softmax = pred_softmax.detach()
			D_out = model_D(pred_softmax)

			generated_label = make_D_label(
				input=D_out,
				value=source_label,
				device=args.device,
				random=True,
			)
			loss_D = bce_loss(D_out, generated_label)
			loss_D = loss_D / args.iter_size / 2
			loss_D.backward()

			loss_D_value += loss_D.item()

			# train with target

			pred_softmax_target = pred_softmax_target.detach()
			D_out = model_D(pred_softmax_target)

			generated_label = make_D_label(
				input=D_out,
				value=target_label,
				device=args.device,
				random=True,
			)

			loss_D = bce_loss(D_out, generated_label)
			loss_D = loss_D / args.iter_size / 2
			loss_D.backward()

			loss_D_value += loss_D.item()

		optimizer_SS.step()
		optimizer_D.step()

		if args.tensorboard:
			scalar_info = {
				'loss_seg_source': loss_seg_source_value,
				'loss_seg_target': loss_seg_target_value,
				'loss_adv_target': loss_adv_target_value,
				'loss_semi_target': loss_semi_target_value,
				'loss_D': loss_D_value,
			}

			for key, val in scalar_info.items():
				writer.add_scalar(key, val, i_iter)

		print('iter = {0:8d}/{1:8d} '
			  'loss_seg_source = {2:.3f} loss_seg_target = {3:.3f} '
			  'loss_adv_target = {4:.3f} '
			  'loss_semi_target = {5:.3f} '
			  'loss_D = {6:.3f}'.format(i_iter, args.total_iterations,
										loss_seg_source_value, loss_seg_target_value,
										loss_adv_target_value,
										loss_semi_target_value,
										loss_D_value))

		every_5_epoch = int((5 * args.total_source) / args.batch_size)
		current_epoch = int(i_iter * args.batch_size / args.total_source)
		if i_iter % every_5_epoch == 0:
			# evaluate on validation
			val_loss_f = validate(
				val_loader=val_loader,
				model=model_SS,
				epoch=current_epoch,
				args=args,
				val_loss=val_loss_f,
			)
			val_loss.append(val_loss_f)
			val_loss_f = np.min(np.array(val_loss))
			if args.tensorboard:
				scalar_info = {
					'loss_val': val_loss_f,
				}
				for key, val in scalar_info.items():
					writer.add_scalar(key, val, i_iter)

			# determine optimal model and save it
			is_better_ss = current_loss_seg < loss_seg_min
			if is_better_ss:
				loss_seg_min = current_loss_seg
				if not os.path.isdir(args.snapshot_dir):
					os.mkdir(args.snapshot_dir)
				torch.save(model_SS.state_dict(), os.path.join(args.snapshot_dir, "modelSS_train_best.pth"))


	current_epoch = int(args.total_iterations * args.batch_size / args.total_source)
	val_loss_f = validate(
		val_loader=val_loader,
		model=model_SS,
		epoch=current_epoch,
		args=args,
		val_loss=val_loss_f,
	)
	val_loss.append(val_loss_f)
	if args.tensorboard:
		scalar_info = {
			'loss_val': val_loss_f,
		}
		for key, val in scalar_info.items():
			writer.add_scalar(key, val, args.total_iterations)

	if args.tensorboard:
		writer.close()

	return val_loss