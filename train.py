import torch
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from dataset import GetLoader
import os
import models 
import time

batch_size=1
lrG = 0.0002
lrD = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_l = 100 

train_list = os.path.join('.','data','cityscapes','train.txt')
val_list = os.path.join('.','data','cityscapes','val.txt')

transform = transforms.Compose([
     transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = GetLoader(
	data_root=os.path.join('.','data','cityscapes','train'),
	data_list= train_list,
	transform1=transform,
	transform2=transform
)

valset = GetLoader(
	data_root=os.path.join('.','data','cityscapes','val'),
	data_list= val_list,
	transform1=transforms.ToTensor(),
	transform2=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = models.Generator(d=64)
D = models.discriminator(d=64)
G.weight_init(0,0.02)
D.weight_init(0,0.02)

G.to(device)
D.to(device)

# titer = iter(trainloader)
# img1,img2 = titer.next()
# print(torch.max(img2))
	
# loss
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
D_optimizer = optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))

epochs = 200

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

torch.cuda.synchronize()

initial_time = time.time()
for epoch in range(epochs):
	start_time = time.time()
	runningGloss = 0
	runningDloss = 0
	count = 0
	for img1,img2 in trainloader:
		
		#discriminator loss
		D.zero_grad()
		
		img1,img2 = img1.to(device),img2.to(device)
		fake_img1 = G(img2)

		D_fake = D(fake_img1,img2).squeeze()
		loss_fake = BCE_loss(D_fake,torch.zeros(D_fake.shape).to(device))

		D_real = D(img1,img2).squeeze()
		loss_real = BCE_loss(D_real,torch.ones(D_real.shape).to(device))

		D_loss = (loss_real+loss_fake)*(0.5)
		runningDloss += D_loss 
		D_loss.backward()
		D_optimizer.step()

		#generator loss
		G.zero_grad()
		fake_img1 = G(img2)
		D_fake = D(fake_img1,img2).squeeze()
		G_loss = BCE_loss(D_fake,torch.ones(D_fake.size()).to(device))+lambda_l*L1_loss(fake_img1,img1)
		runningGloss += G_loss
		G_loss.backward()
		G_optimizer.step()
		count+=1
		if(count%10==0):
			print("Progress of epoch {}: {}".format(epoch+1,count*100/len(trainloader)),end='\r')


	runningGloss /= len(trainloader)
	runningDloss /= len(trainloader)
	train_hist['D_losses'].append(runningDloss)
	train_hist['G_losses'].append(runningGloss)
	
	if((epoch+1)%10 == 0):
		torch.save({
			'epoch':epoch+1,
			'G_state_dict' : G.state_dict(),
			'D_state_dict' : D.state_dict(),
			'D_opt_state_dict' : D_optimizer.state_dict(),
			'G_opt_state_dict' : G_optimizer.state_dict(),
			},os.path.join('.','checkpoint',str(epoch+1)+".pth"))


	print("Epoch: {}".format(epoch+1))
	print("Epoch time: {}".format(start_time-time.time()))
	print("Total_time: {}".format(initial_time-time.time()))
	print("D_loss: {}".format(runningDloss))
	print("G_loss: {}".format(runningGloss))



