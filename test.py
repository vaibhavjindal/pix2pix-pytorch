import torch
import numpy as np 
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from dataset_test import GetLoader
import os
import models 

batch_size=1

model_path =os.path.join('.','checkpoint','10.pth')

val_list = os.path.join('.','data','cityscapes','val.txt')

transform = transforms.Compose([
     transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

valset = GetLoader(
	data_root=os.path.join('.','data','cityscapes','val'),
	data_list= val_list,
	transform1=transforms.ToTensor(),
	transform2=transforms.ToTensor()
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4)

model = models.Generator(d=64)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['G_state_dict'])

model.train()
img_paths = valset.img_paths

i=0
for img1,img2 in valloader:
	fake_img1 = model(img2)
	torchvision.utils.save_image(fake_img1,os.path.join('.','results','10',str(img_paths[i])))
	i+=1
	print(i)


