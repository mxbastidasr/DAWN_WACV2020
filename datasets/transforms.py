import torch
from torchvision import transforms
import numpy as np

class GCN(object):
    """Global constrast normalization
    """

    def __init__(self,
                 channel_wise=True,
                 scale=1.,
                 subtract_mean=True, use_std=True, sqrt_bias=10., min_divisor=1e-8):
        self.scale = scale
        self.subtract_mean = subtract_mean
        self.use_std = use_std
        self.sqrt_bias = sqrt_bias
        self.min_divisor = min_divisor
        self.channel_wise = channel_wise

    def __call__(self, img):
        if self.channel_wise:
            assert(img.shape[0] == 3)
            for i in range(3):  # RGB images
                if self.subtract_mean:
                    img[i, :, :] = img[i, :, :] - torch.mean(img[i, :, :])
                if self.use_std:
                    norm = torch.sqrt(
                        self.sqrt_bias + torch.var(img[i, :, :])) / self.scale
                else:
                    norm = torch.sqrt(
                        self.sqrt_bias + torch.sum(torch.pow(img[i, :, :], 2))) / self.scale
                img[i, :, :] = img[i, :, :] / norm
            norm[norm < self.min_divisor] = 1.
            return img
        else:
            if self.subtract_mean:
                mean = torch.mean(img)
                img = img - mean
            if self.use_std:
                norm = torch.sqrt(self.sqrt_bias + torch.var(img)) / self.scale
            else:
                norm = torch.sqrt(
                    self.sqrt_bias + torch.sum(torch.pow(img, 2))) / self.scale
            norm[norm < self.min_divisor] = 1.
            img = img / norm
            return img

    def __repr__(self):
        return self.__class__.__name__



class Lighting(object):
	"""Lighting noise(AlexNet - style PCA - based noise)"""

	def __init__(self, alphastd, eigval, eigvec):
		self.alphastd = alphastd
		self.eigval = eigval
		self.eigvec = eigvec

	def __call__(self, img):
		if self.alphastd == 0:
			return img

		alpha = img.new().resize_(3).normal_(0, self.alphastd)
		rgb = self.eigvec.type_as(img).clone()\
			.mul(alpha.view(1, 3).expand(3, 3))\
			.mul(self.eigval.view(1, 3).expand(3, 3))\
			.sum(1).squeeze()

		return img.add(rgb.view(3, 1, 1).expand_as(img))


class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
			# The normalize code -> t.sub_(m).div_(s)
		return tensor

if __name__ == "__main__":
    from PIL import Image
    img = Image.open("data/baboon.png")

    transform_train = transforms.Compose([
        # Allow random zoom (50% max)
        # TODO: Check the best auguementation value in this case
        transforms.RandomResizedCrop(
            224, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        GCN(),

    ])
    img_tensor = transform_train(img)
    img_tensor_min = img_tensor.min()
    img_tensor_max = img_tensor.max()
    print(img_tensor_min, img_tensor_max, torch.mean(img_tensor))
    img = transforms.ToPILImage()(img_tensor)
    img.show()

    img_tensor -= img_tensor_min
    img_tensor /= (img_tensor_max - img_tensor_min)
    img = transforms.ToPILImage()(img_tensor)
    img.show()
