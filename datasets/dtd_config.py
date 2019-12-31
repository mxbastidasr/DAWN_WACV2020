import torch.utils.data as data

from PIL import Image
import os
import os.path
import re

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	imlabel=-1
	n=''
	n_=''
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath = line.strip().split()
			n_=re.split(r'/',impath[0])[0] #obtain labels
			if(n!=n_):
				imlabel+=1
				n=n_

			imlist.append( (impath[0], imlabel) )
					
	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		
		self.imlist = flist_reader(root)
		self.root=re.split('labels/', root)[0]
		
		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root+'dtd/',impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imlist)