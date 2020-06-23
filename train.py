# Global imports
import argparse
import os
import shutil
import time
import sys
import math
import time
import numpy as np

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Models imports
import models.densenet as dn
import models.vgg as vgg
import models.wcnn as wcnn
import models.dawn as dawn
import models.resnet as resnet

# Other imports (datasets)
from datasets.dtd_config import ImageFilelist
from datasets.transforms import GCN, UnNormalize, Lighting
from torch.utils.data.sampler import SubsetRandomSampler

# used for logging to TensorBoard
try:
	from tensorboard_logger import configure, log_value
except:
	print("Ignore tensorboard logger")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--drop', default=10, type=int, 
					help='drop learning rate')
parser.add_argument('--epochs', default=300, type=int,
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
					help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
					help='name of experiment')
parser.add_argument('--tensorboard',
					help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--multigpu',
					help='Do the training using multiple GPU', action='store_true')
parser.add_argument(
	'--database', help='database used for training', choices=['cifar-10', 'cifar-100','stl','svhn', 'kth','kth-mix','dtd', 'imagenet','imagenet_ECCV'])
parser.add_argument('--num_clases', default=11, type=int,
					help='num of classes, only works for kth dataset')
parser.add_argument(
	'--summary', default=False, action='store_true')
parser.add_argument(
	'--gcn', default=False, action='store_true')
parser.add_argument(
	"--lrdecay", nargs='+', type=int
)
parser.add_argument("--tempdir", type=str, default='/home/agruson/scratch/data',
					help='where to put the dataset if we need to download')

parser.add_argument('--split_data', default=0, type=int,
					help='take a limited dataset for training')

# ----------- Special option for KTH
# As for KTH, the test and validation set are splitted
# into multiple directory
parser.add_argument('--traindir', default='dataset/kth/train',
					type=str, help='KTH image dir for training')
parser.add_argument('--valdir', default='dataset/kth/test',
					type=str, help='KTH image dir for validation')

# ---- Monkey test
parser.add_argument('--monkey', default=False, action="store_true")

# ----- Pretrained imageNet
parser.add_argument('--pretrained', default=False, action="store_true")

# ----------- Models
subparsers = parser.add_subparsers(dest="model")

# Densenet options
parser_densenet = subparsers.add_parser('densenet')
parser_densenet.add_argument('--layers', default=100, type=int,
							 help='total number of layers (default: 100)')
parser_densenet.add_argument('--growth', default=12, type=int,
							 help='number of new channels per layer (default: 12)')
parser_densenet.add_argument('--droprate', default=0, type=float,
							 help='dropout probability (default: 0.0)')
# TODO: Reduce is 0.5 in the original code
# TODO: Need to check this part
parser_densenet.add_argument('--reduce', default=1.0, type=float,
							 help='compression rate in transition stage (default: 1.0)')
parser_densenet.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
							 help='To not use bottleneck block')
parser_densenet.set_defaults(bottleneck=True)
parser_densenet.add_argument('--no_init_conv', default=False, action='store_true')

# WCNN options
# Note that shinnet might be trained with lifting scheme
parser_wcnn = subparsers.add_parser('wcnn')
parser_wcnn.add_argument("--wavelet", choices=['haar', 'db2', 'lifting'])
parser_wcnn.add_argument("--levels", default=4, type=int)

# DAWN options
parser_dwnn = subparsers.add_parser('dawn')
parser_dwnn.add_argument("--regu_details", default=0.1, type=float)
parser_dwnn.add_argument("--regu_approx", default=0.1, type=float)
parser_dwnn.add_argument("--levels", default=4, type=int)
parser_dwnn.add_argument("--first_conv", default=32, type=int)
parser_dwnn.add_argument(
	"--classifier", default='mode1', choices=['mode1', 'mode2','mode3'])
parser_dwnn.add_argument(
	"--kernel_size", type=int, default=3
)
parser_dwnn.add_argument(
	"--no_bootleneck", default=False, action='store_true'
)
parser_dwnn.add_argument(
	"--share_weights", default=False, action='store_true'
)
parser_dwnn.add_argument(
	"--simple_lifting", default=False, action='store_true'
)
parser_dwnn.add_argument(
	"--haar_wavelet", default=False, action='store_true'
)
parser_dwnn.add_argument(
	'--warmup', default=False, action='store_true'
)

# Resnet options
parser_resnet = subparsers.add_parser('resnet')
parser_resnet.add_argument("--use_normal", default=False, action='store_true',
						   help='use normal resnet for CIFAR (described inside the paper)')
parser_resnet.add_argument("--size_normal", default=3,
						   type=int, help="the size used for normal resnet")
parser_resnet.add_argument(
	"--levels", default=4, type=int, help="number of levels for resnet (3 or 4)")

# scatter options
parser_scatter = subparsers.add_parser('scatter')
parser_scatter.add_argument('--scat', default=2, type=int,
						help='scattering scale, j=0 means no scattering')
parser_scatter.add_argument('--N', default=32, type=int,
						help='size of the crop')
parser_scatter.add_argument('--classifier', type=str, default='WRN',help='classifier model [WRN, mlp, linear]')
parser_scatter.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
parser_scatter.add_argument('--blocks', type=int, default=2,help='for WRN number of blocks of layers: n ')
parser_scatter.add_argument('--use_avg_pool', default=False, action='store_true', help='use avg pooling before the classifier')

# VGG options
parser_vgg = subparsers.add_parser('vgg')
best_prec1 = 0


def main():
	global args, best_prec1
	args = parser.parse_args()
	if args.tensorboard:
		configure("runs/%s" % (args.name))
	print("Launch...")
	print(sys.argv)
	
	# Global configuration of the datasets
	USE_COLOR = not args.monkey
	kwargs = {'num_workers': 4, 'pin_memory': True}

	##############################
	# Database loading
	##############################
	# Data loading code
	# TODO: For now only support 224x224 input size of 32x32. Need more work to support more resolutions
	if args.database == 'cifar-10':
		if not USE_COLOR:
			raise "CIFAR-10 does not handle training with gray images"
		# Data augumentation
		normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
										 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
		
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])
		
		# CIFAR-10
		data_CIFAR10=datasets.CIFAR10(args.tempdir, train=True, download=True,
								transform=transform_train)
		if args.split_data>0:
			sampler=torch.utils.data.sampler.WeightedRandomSampler(weights=[1] * 10000, num_samples=args.split_data)
			shuffle=False
		else:
			sampler=None
			shuffle=True

		train_loader = torch.utils.data.DataLoader(
				data_CIFAR10,
				batch_size=args.batch_size, shuffle=shuffle,sampler=sampler, **kwargs)
		
		val_loader = torch.utils.data.DataLoader(
				datasets.CIFAR10(args.tempdir,
								train=False, transform=transform_test),
				batch_size=args.batch_size, shuffle=True, **kwargs)
		
		NUM_CLASS = 10
		INPUT_SIZE = 32
	elif args.database == 'cifar-100':
		if not USE_COLOR:
			raise "CIFAR-100 does not handle training with gray images"
		# Data augumentation
		# From: https://github.com/meliketoy/wide-resnet.pytorch/blob/master/config.py
		normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
										 std=[0.2675, 0.2565, 0.2761])
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])
		data_CIFAR100=datasets.CIFAR100(args.tempdir, train=True, download=True,
								transform=transform_train)
		if args.split_data>0:
			sampler=torch.utils.data.sampler.WeightedRandomSampler(weights=[1] * 10000, num_samples=args.split_data)
			shuffle=False
		else:
			sampler=None
			shuffle=True

		# CIFAR-100
		train_loader = torch.utils.data.DataLoader(
				data_CIFAR100,
				batch_size=args.batch_size, shuffle=shuffle,sampler=sampler, **kwargs)
		
		val_loader = torch.utils.data.DataLoader(
				datasets.CIFAR100(args.tempdir,
								train=False, transform=transform_test),
				batch_size=args.batch_size, shuffle=True, **kwargs)
		
		NUM_CLASS = 100
		INPUT_SIZE = 32
	elif args.database == 'kth':
		if not USE_COLOR and args.gcn:
			raise "It is not possible to use grayimage and GCN"
		# Data auguementation
		if args.gcn:
			normalize = GCN()
		else:
			 # TODO: Use the same normalization than CIFAR-10
			 # TODO: That might be suboptimal...
			if USE_COLOR:
				normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
												 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
			else:
				normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3]],
												 std=[x/255.0 for x in [63.0]])

		add_transform = []
		if not USE_COLOR:
			add_transform += [transforms.Grayscale(num_output_channels=1)]

		transform_train = transforms.Compose(
			add_transform + [transforms.Resize((256, 256)),
							 transforms.RandomCrop(224),
							 transforms.RandomHorizontalFlip(),
							 transforms.RandomVerticalFlip(),
							 transforms.ToTensor(),
							 normalize
							 ])
		transform_test = transforms.Compose(
			add_transform + [transforms.Resize((256, 256)),
							 transforms.CenterCrop((224, 224)),
							 transforms.ToTensor(),
							 normalize
							 ])

		kth_train_dataset = datasets.ImageFolder(root=args.traindir,
												 transform=transform_train)
		
		kth_test_dataset = datasets.ImageFolder(root=args.valdir,
												transform=transform_test)
		train_loader = torch.utils.data.DataLoader(kth_train_dataset, shuffle=True,
												   batch_size=args.batch_size, **kwargs)

		val_loader = torch.utils.data.DataLoader(kth_test_dataset, shuffle=True,
												 batch_size=args.batch_size, **kwargs)
		NUM_CLASS = args.num_clases
		INPUT_SIZE = 224

	elif args.database == 'imagenet_ECCV':
		
		if not USE_COLOR:
			raise "Imagenet does not handle training with gray images"

		_IMAGENET_PCA = {
			'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
			'eigvec': torch.Tensor([
				[-0.5675,  0.7192,  0.4009],
				[-0.5808, -0.0045, -0.8140],
				[-0.5836, -0.6948,  0.4203],
			])
		}

		if args.gcn:
			normalize = GCN()
		else:
			normalize = transforms.Normalize(mean=[x for x in [0.485, 0.456, 0.406]],
											 std=[x for x in [0.229, 0.224, 0.225]])

		transform_train = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.RandomCrop(224),
			transforms.ColorJitter(
				brightness=0.4, contrast=0.4, saturation=0.4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
			normalize
		])
		transform_test = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.CenterCrop((224, 224)),
			transforms.ToTensor(),
			normalize
		])

		train_dataset = datasets.ImageFolder(root=args.traindir,
												 transform=transform_train)
		test_dataset = datasets.ImageFolder(root=args.valdir,
												transform=transform_test)
		train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
												   batch_size=args.batch_size, **kwargs)

		val_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,
												 batch_size=args.batch_size, **kwargs)
		NUM_CLASS = 1000
		INPUT_SIZE = 224

	else:
		raise "Unknown database"

	if not USE_COLOR and not args.model == "dawn":
		raise "Only DAWN support training with gray images"

	##################################
	# Create model
	##################################
	# TODO: This condition only work if we have only two image size
	big_input = INPUT_SIZE != 32
	# Only for scattering transform (as coefficient are precomptuted)
	scattering=None 
	if args.model == 'densenet':
		no_init_conv = args.no_init_conv
		if(INPUT_SIZE > 128):
			# For these input size, init conv are necessary
			no_init_conv = False 
		model = dn.DenseNet3(args.layers, NUM_CLASS, args.growth, reduction=args.reduce,
							 bottleneck=args.bottleneck, dropRate=args.droprate, init_conv=not no_init_conv)
	elif args.model == 'vgg':
		model = vgg.VGG(NUM_CLASS, big_input=big_input)
	elif args.model == 'wcnn':
		model = wcnn.WCNN(
			NUM_CLASS, big_input=big_input, wavelet=args.wavelet, levels=args.levels)
	elif args.model == 'dawn':
		# Our model
		model = dawn.DAWN(NUM_CLASS, big_input=big_input,
						  first_conv=args.first_conv,
						  number_levels=args.levels,
						  kernel_size=args.kernel_size,
						  no_bootleneck=args.no_bootleneck,
						  classifier=args.classifier,
						  share_weights=args.share_weights,
						  simple_lifting=args.simple_lifting,
						  COLOR=USE_COLOR,
						  regu_details=args.regu_details,
						  regu_approx=args.regu_approx,
						  haar_wavelet=args.haar_wavelet
						  )
	elif args.model=='scatter':
		from kymatio import Scattering2D
		from models.scatter.Scatter_WRN import Scattering2dCNN, ScatResNet
		
		if(INPUT_SIZE == 224):
			# KTH
			scattering = Scattering2D(J=args.scat, shape=(args.N, args.N), max_order=args.mode)
			scattering = scattering.cuda()
			model = ScatResNet(args.scat, INPUT_SIZE, NUM_CLASS, args.classifier, args.mode)
		else:
			# Precomputation
			scattering = Scattering2D(J=args.scat, shape=(args.N, args.N), max_order=args.mode)
			scattering = scattering.cuda()
			model = Scattering2dCNN(args.classifier,J=args.scat,N=args.N,
									num_classes=NUM_CLASS,blocks=args.blocks,
									mode=args.mode, use_avg_pool=args.use_avg_pool)
	elif args.model == 'resnet':
		if big_input:
			import torchvision
			model = torchvision.models.resnet18(pretrained=args.pretrained)
			model.fc = nn.Linear(512, NUM_CLASS)
		else:
			if args.use_normal:
				model = resnet.ResNetCIFARNormal(
					[args.size_normal, args.size_normal, args.size_normal],
					num_classes=NUM_CLASS)
			else:
				model = resnet.ResNetCIFAR([2, 2, 2, 2],
										   num_classes=NUM_CLASS, levels=args.levels)
	else:
		raise "Unknown model"

	# get the number of model parameters
	print("Number of model parameters            : {:,}".format(
		sum([p.data.nelement() for p in model.parameters()])))
	print("Number of *trainable* model parameters: {:,}".format(
		sum(p.numel() for p in model.parameters() if p.requires_grad)))

	# for training on multiple GPUs.
	# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
	if args.multigpu:
		model = torch.nn.DataParallel(model).cuda()
	else:
		model = model.cuda()

	# Print network summary
	if args.summary:
		# For display like Keras
		from torchsummary import summary
		summary(model, input_size=(3, INPUT_SIZE, INPUT_SIZE))

	# CSV files statistics
	csv_logger = CSVStats()

	if args.monkey:
		# This is a special condition
		# Only to test and visualize the multi-resolution
		# output
		f = open('./data/baboon.png', 'rb')
		from PIL import Image
		img = Image.open(f)
		img.convert('RGB')
		img.show()

		# Do the transformation
		to_pil = transforms.ToPILImage()
		if USE_COLOR:
			unormalize = UnNormalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
		else:
			unormalize = UnNormalize(mean=[x/255.0 for x in [125.3]],
									 std=[x/255.0 for x in [63.0]])
		tensor_trans = transform_test(img)
		img_trans = to_pil(torch.clamp(unormalize(tensor_trans),0.0, 1.0))
		img_trans.show()
		img_trans.save("trans.png")

		# Make pytorch with the batch size
		tensor_trans = tensor_trans[None, :, :, :]

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			
			if args.monkey:
				dump_image(tensor_trans, args.start_epoch, model, unormalize)
				raise "stop here!"
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# define loss function (criterion) and pptimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								nesterov=True,
								weight_decay=args.weight_decay)

	if args.model == 'dawn':
		if args.warmup:
			model.initialization = True
			train_init(train_loader, model, criterion, optimizer)
			model.initialization = False # Switch

	for epoch in range(args.start_epoch, args.epochs):
		t0 = time.time()
		adjust_learning_rate(optimizer, epoch, args.drop)
		# TODO: Clean this code
		if args.model=='dawn':
			args_model='dawn'
		
		#model, optimizer = amp.initialize(model, optimizer)
		# train for one epoch        
		prec1_train, prec5_train, loss_train = train(
			train_loader, model, criterion, optimizer, epoch, args.model == args_model,scattering=scattering)

		# Optimize cache on cuda
		# This is quite important to avoid memory overflow
		# when training big models. The performance impact
		# seems minimal
		torch.cuda.empty_cache()

		# evaluate on validation set
		prec1_val, prec5_val, loss_val = validate(
			val_loader, model, criterion, epoch,  args.model == args_model,scattering=scattering)

		if args.monkey:
			# In this case, we will output the Monkey image
			dump_image(tensor_trans, epoch, model,unormalize)
				
		# Print some statistics inside CSV
		csv_logger.add(prec1_train, prec1_val, prec5_train, prec5_val, loss_train, loss_val)

		# remember best prec@1 and save checkpoint
		is_best = prec1_val > best_prec1
		best_prec1 = max(prec1_val, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
		}, is_best)
		csv_logger.write()

		# Final print
		print(' * Train[{:.3f} %, {:.3f} %, {:.3f} loss] Val [{:.3f} %, {:.3f}%, {:.3f} loss] Best: {:.3f} %'.format(
			prec1_train, prec5_train, loss_train, prec1_val, prec5_val, loss_val, best_prec1))
		print('Time for', epoch, "/", args.epochs, time.time() - t0)

	print('Best accuracy: ', best_prec1)


def dump_image(tensor_trans, epoch, model, unormalize):
	from PIL import Image
	to_pil = transforms.ToPILImage()
	with torch.no_grad():
		input = tensor_trans.cuda()
		input_var = torch.autograd.Variable(input)
		out = model.image_levels(input)

		# (LL, LH, HL, HH)
		#levels = 0
		for levels in range(4):
			out_LL = to_pil(unormalize(out[levels][0][0,:,:,:].cpu()))
			
			# Signal is centred in the middle (due to the unormalized)
			out_LH = to_pil(torch.abs(out[levels][1][0,:,:,:].cpu())*3)
			out_HL = to_pil(torch.abs(out[levels][2][0,:,:,:].cpu())*3)
			out_HH = to_pil(torch.abs(out[levels][3][0,:,:,:].cpu())*3)
			
			# Compose the image
			size = out_LL.size[0]*2
			new_im = Image.new('RGB', (size, size))
			new_im.paste(out_LL, (0,0))
			new_im.paste(out_LH, (0 + size // 2,0))
			new_im.paste(out_HL, (0, size // 2))
			new_im.paste(out_HH, (0 + size // 2, size // 2))
			new_im.save("epoch_{}_level_{}.png".format(epoch,levels))

def train_init(train_loader, model, criterion, optimizer):
	"""Train for one epoch on the training set"""
	end = time.time()
	# switch to train mode
	model.train()
	for epoch in range(3):
		batch_time = AverageMeter()
		losses_total = AverageMeter()
		losses_class = AverageMeter()
		losses_regu = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		for i, (input, target) in enumerate(train_loader):
			if i == 0 and epoch == 0:
				save_input = input[0,:,:,:]
			target = target.cuda()
			input = input.cuda()
			
			input_var = torch.autograd.Variable(input)
			target_var = torch.autograd.Variable(target)

			# compute output
			is_regu_activated = False
			diffs, regus = model(input_var)
			loss_regu = sum(regus)
			loss_class = sum(diffs)
			
			loss_total = loss_regu + loss_class
			
			# measure accuracy and record loss
			losses_total.update(loss_total.item(), input.size(0))
			losses_regu.update(loss_regu.item(), input.size(0))
			losses_class.update(loss_class.item(), input.size(0))
		
			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss_total.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})\t'
					'Loss (Regu) {loss_regu.val:.4f} ({loss_regu.avg:.4f})'.format(
						epoch, i, len(train_loader), batch_time=batch_time,
						loss_class=losses_class, loss_regu=losses_regu))

		# Do the inference
		print(save_input)
		model.eval()
		inp = save_input[None, :, :, :]
		inp = inp.cuda()
		print(inp.shape)
		diffs, regus = model(inp)
		for d in diffs:
			print(d.mean())
		
		print(save_input.mean())

		from PIL import Image
		to_pil = transforms.ToPILImage()
		out, out_d = model.process_levels(inp)
		new_im = Image.new('RGB', (96, 96))
		new_im.paste(to_pil(save_input))
		new_im.save("epoch_{}_input.png".format(epoch))

		for i in range(len(out)):
			new_im = Image.new('RGB', (out[i].shape[2]*2, out[i].shape[3]))
			out_LL = to_pil(out[i][0,:,:,:].cpu())
			out_down_LL = to_pil(out_d[i][0,:,:,:].cpu())
			new_im.paste(out_LL, (0,0))
			new_im.paste(out_down_LL, (0 + out[i].shape[2],0))
			new_im.save("epoch_{}_level_{}.png".format(epoch,i))

		print(' * Train[{:.3f} loss]'.format(losses_total.avg))
		

	return losses_total.avg


def train(train_loader, model, criterion, optimizer, epoch, is_dwnn,scattering=None):
	"""Train for one epoch on the training set"""
	batch_time = AverageMeter()
	losses_total = AverageMeter()
	losses_class = AverageMeter()
	losses_regu = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	for i, (input, target) in enumerate(train_loader):

		target = target.cuda()
		input = input.cuda()
		
		if args.model=='scatter':
			input = scattering(input)
			
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		# compute output
		# Note that this part is to handle the regularisation terms
		# In this code version, the loss is augmented by these
		# regularisation terms
		is_regu_activated = False
		if is_dwnn:
			output, regus = model(input_var)
			loss_class = criterion(output, target_var)
			loss_total = loss_class
			# If no regularisation used, None inside regus
			if regus[0]:
				loss_regu = sum(regus)
				loss_total += loss_regu
				is_regu_activated = True
			
		else:
			output = model(input_var)
			loss_class = criterion(output, target_var)
			loss_total = loss_class
	
		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		if args.num_clases>=5:
			p_m=5
		else:
			p_m=3
		prec5 = accuracy(output.data, target, topk=(p_m,))[0]#5
		losses_total.update(loss_total.item(), input.size(0))

		if is_regu_activated:
			losses_regu.update(loss_regu.item(), input.size(0))
		else:
			losses_regu.update(0.0, input.size(0))
		losses_class.update(loss_class.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})\t'
				  'Loss (Regu) {loss_regu.val:.4f} ({loss_regu.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  loss_class=losses_class, loss_regu=losses_regu, 
					  top1=top1,top5=top5))

	# log to TensorBoard
	if args.tensorboard:
		log_value('train_loss', losses_total.avg, epoch)
		log_value('train_acc', top1.avg, epoch)

	return (top1.avg, top5.avg, losses_total.avg)


def validate(val_loader, model, criterion, epoch, is_dwnn, scattering=None):
	"""Perform validation on the validation set"""
	batch_time = AverageMeter()
	losses_total = AverageMeter()
	losses_class = AverageMeter()
	losses_regu = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			target = target.cuda()
			input = input.cuda()
			if args.model=='scatter':					
				input = scattering(input)

			input_var = torch.autograd.Variable(input)
			target_var = torch.autograd.Variable(target)
			
			# compute output
			is_regu_activated = False
			if is_dwnn:
				output, regus = model(input_var)
				loss_class = criterion(output, target_var)
				loss_total = loss_class
				# If no regularisation used, None inside regus
				if regus[0]:
					loss_regu = sum(regus)
					loss_total += loss_regu
					is_regu_activated = True
			else:
				output = model(input_var)
				loss_class = criterion(output, target_var)
				loss_total = loss_class

			# measure accuracy and record loss
			prec1 = accuracy(output.data, target, topk=(1,))[0]
			if args.num_clases>=5:
				p_m=5
			else:
				p_m=3
			prec5 = accuracy(output.data, target, topk=(p_m,))[0]
			losses_total.update(loss_total.item(), input.size(0))
			if is_regu_activated:
				losses_regu.update(loss_regu.item(), input.size(0))
			else:
				losses_regu.update(0.0, input.size(0))
			losses_class.update(loss_class.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})\t'
					  'Loss (Regu) {loss_regu.val:.4f} ({loss_regu.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
						  i, len(val_loader), batch_time=batch_time,
						  loss_class=losses_class, loss_regu=losses_regu, 
						  top1=top1, top5=top5))

	# log to TensorBoard
	if args.tensorboard:
		log_value('val_loss', losses_total.avg, epoch)
		log_value('val_acc', top1.avg, epoch)
	return (top1.avg, top5.avg, losses_total.avg)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	"""Saves checkpoint to disk"""
	
	directory = "runs/%s/" % (args.name)

	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	
	if is_best:

		best_name=directory + 'model_best.pth.tar'
		#shutil.copyfile(filename, best_name)
		torch.save(state, best_name)
		


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class CSVStats(object):
	def __init__(self):
		self.prec1_train = []
		self.prec1_val = []
		self.prec5_train = []
		self.prec5_val = []
		self.loss_train = []
		self.loss_val = []

	def add(self, p1_train, p1_val, p5_train, p5_val, l_train, l_val):
		self.prec1_train.append(p1_train)
		self.prec1_val.append(p1_val)
		self.prec5_train.append(p5_train)
		self.prec5_val.append(p5_val)
		self.loss_train.append(l_train)
		self.loss_val.append(l_val)

	def write(self):
		out = "runs/%s/stats.csv" % (args.name)
		with open(out, "w") as f:
			f.write('prec1_train,prec1_val,prec5_train,prec5_val,loss_train,loss_val\n')
			for i in range(len(self.prec1_val)):
				f.write("{:.5f},{:.5f},{:.5f},{:.5f},{},{}\n".format(
					self.prec1_train[i], self.prec1_val[i],
					self.prec5_train[i], self.prec5_val[i],
					self.loss_train[i], self.loss_val[i]))

	def read(self, out):
		raise "Unimplemented"


def adjust_learning_rate(optimizer, epoch, inv_drop):
	"""Sets the learning rate to the initial LR decayed
	LR decayed is controlled by the user."""
	lr = args.lr
	drop = 1.0 / inv_drop
	factor = math.pow(
		drop, sum([1.0 if epoch >= e else 0.0 for e in args.lrdecay]))
	lr = lr * factor
	print("Learning rate: ", lr)
	# log to TensorBoard
	if args.tensorboard:
		log_value('learning_rate', lr, epoch)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


if __name__ == '__main__':
	main()
