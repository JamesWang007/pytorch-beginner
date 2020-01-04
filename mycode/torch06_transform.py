# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:51:46 2019
link : https://www.pytorchtutorial.com/docs/torchvision/torchvision-transform/

@author: bejin
"""

'''
# pytorch torchvision transform
# 对PIL.Image进行变换
class torchvision.transforms.Compose(transforms)
class torchvision.transforms.CenterCrop(size)
class torchvision.transforms.RandomCrop(size, padding=0)

class torchvision.transforms.RandomHorizontalFlip
class torchvision.transforms.RandomSizedCrop(size, interpolation=2)
class torchvision.transforms.Pad(padding, fill=0)

# 对Tensor进行变换
class torchvision.transforms.Normalize(mean, std)

# Conversion Transforms
class torchvision.transforms.ToTensor
class torchvision.transforms.ToPILImage

# 通用变换
class torchvision.transforms.Lambda(lambd)

'''


from torchvision import transforms
from PIL import Image
crop = transforms.Scale(12)
img = Image.open('test.jpg')








