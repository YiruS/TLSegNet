from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import os
import sys

from tensorboardX import SummaryWriter
# from torchviz import make_dot
from graphviz import Digraph

import utils.photometric_transforms as ph_transforms
from helper_functions.config import ReadConfig_GAN, augment_args_GAN
from models.model_utils import load_models

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def make_dot(var, params):
	""" Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
	param_map = {id(v): k for k, v in params.items()}
	print(param_map)

	node_attr = dict(style='filled',
					 shape='box',
					 align='left',
					 fontsize='12',
					 ranksep='0.1',
					 height='0.2')
	dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
	seen = set()

	def size_to_str(size):
		return '(' + (', ').join(['%d' % v for v in size]) + ')'

	def add_nodes(var):
		if var not in seen:
			if torch.is_tensor(var):
				dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
			elif hasattr(var, 'variable'):
				u = var.variable
				node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
				dot.node(str(id(var)), node_name, fillcolor='lightblue')
			else:
				dot.node(str(id(var)), str(type(var).__name__))
			seen.add(var)
			if hasattr(var, 'next_functions'):
				for u in var.next_functions:
					if u[0] is not None:
						dot.edge(str(id(u[0])), str(id(var)))
						add_nodes(u[0])
			if hasattr(var, 'saved_tensors'):
				for t in var.saved_tensors:
					dot.edge(str(id(t)), str(id(var)))
					add_nodes(t)

	add_nodes(var.grad_fn)
	return dot

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

	(args, opts) = parser.parse_args()

	if args.ini_file is None:
		print ('Model config file required, quitting...')
		sys.exit(0)

	if args.tensorboard:
		if not os.path.exists(args.log_dir):
			os.makedirs(args.log_dir)

		writer = SummaryWriter(args.log_dir)
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
	).to(args.device)
	# model_D1 = load_models(
	# 	mode = "Discriminator",
	# 	device = device,
	# 	args = args,
	# )
	model_D = load_models(
		mode="Discriminator",
		device=device,
		args=args,
	).to(args.device)

	dummy_input = torch.zeros(1, 1, 184, 184, dtype=torch.float, requires_grad=False).to(args.device)
	seg_out, seg_softmax = model_SS(Variable(dummy_input))
	# g = make_dot(seg_out, model_SS.state_dict())
	# g.view()

	d_output = model_D(seg_softmax)
	d_Net = make_dot(d_output, model_D.state_dict())
	d_Net.view()



