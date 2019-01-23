import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import cv2
from loss import *

import numpy as np
import cv2
import random
import time
import os
import argparse

from models import *
from func import *
from data.dataset_util import RainDataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

class trainer:
	def __init__(self, opt):

		self.net_D = Discriminator().cuda() 
		self.net_G = Generator().cuda()
		self.optim1 = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_G.parameters()), lr = opt.lr, betas = (0.5,0.99))
		self.optim2 = torch.optim.Adam(filter(lambda p : p.requires_grad, self.net_D.parameters()), lr = opt.lr, betas = (0.5,0.99))
		self.start = opt.load
		self.iter = opt.iter
		self.batch_size = opt.batch_size
		train_dataset = RainDataset(opt)
		valid_dataset = RainDataset(opt, is_eval=True)
		train_size = len(train_dataset)
		valid_size = len(valid_dataset)
		self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
		self.valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size)

		print("# train set : {}".format(train_size))
		print("# eval set : {}".format(valid_size))

		self.expr_dir = opt.checkpoint_dir

		#Attention Loss
		self.criterionAtt = AttentionLoss(theta=0.8, iteration=4)
		#GAN Loss
		self.criterionGAN = GANLoss(real_label=1.0, fake_label=0.0)
		#Perceptual Loss
		self.criterionPL = PerceptualLoss()
		#Multiscale Loss
		self.criterionML = MultiscaleLoss(ld = [0.6,0.8,1.0],batch=self.batch_size)
		#MAP Loss
		self.criterionMAP = MAPLoss(gamma = 0.05)
		#MSE Loss
		self.criterionMSE = nn.MSELoss().cuda()
		
		self.out_path = './weight/'
	def forward_process(self,I_,GT, is_train=True):
		M_ = []
		for i in range(I_.shape[0]):
			M_.append(get_mask(np.array(I_[i]),np.array(GT[i])))
		M_ = np.array(M_)

		M_ = torch_variable(M_, is_train)
		I_ = torch_variable(I_, is_train)
		GT_ = torch_variable(GT, is_train)
		
		A_, t1, t2, t3 = self.net_G(I_)
		# print 'mask len', len(A_)
		S_ = [t1,t2,t3]
		O_ = t3

		loss = self.criterionMSE(O_,GT_.detach())

		if is_train:
			#attention_loss		
			loss_att = self.criterionAtt(A_,M_.detach())

			#perceptual_loss O_: generation, T_: GT
			loss_PL = self.criterionPL(O_, GT_.detach())

			#Multiscale_loss
			loss_ML = self.criterionML(S_,GT)

			# print('t3', t3.shape)
			# D(Fake)

			D_map_O, D_fake = self.net_D(t3)
			# D(Real)
			# GT = torch_variable(GT,is_train, is_grad=True)
			D_map_R, D_real = self.net_D(GT_)

			loss_MAP = self.criterionMAP(D_map_O, D_map_R, A_[-1].detach())
			# 1 - D_real
			# 0 - D_fake
			# loss_GAN_fake = self.criterionGAN(D_fake,is_real=False)
			# loss_GAN_real = self.criterionGAN(D_real,is_real=True)
			# loss_gen_D = torch.log(1.0-loss_GAN_fake)
			loss_fake = self.criterionGAN(D_fake,is_real=False) # BCE 1, D_fake -(log(1-fake))
			loss_real = self.criterionGAN(D_real,is_real=True) # BCE 0, D_real -log(real)
			#D_real, 1
			loss_D = loss_real+loss_fake + loss_MAP
			# print (loss_gen_D), (loss_att), (loss_ML), (loss_PL) 
			loss_G = 0.01 * (-loss_fake) + loss_att + loss_ML + loss_PL

			output = [loss_G, loss_D, loss_PL, loss_ML, loss_att, loss_MAP, loss]
		else: # validation
			output = loss

		return output

	def train_start(self):
		loss_sum = 0.
		valid_loss_sum = 0.
		# I_: input raindrop image
		# A_: attention map(Mask_list) from ConvLSTM
		# M_: mask GT
		# O_: output image of the autoencoder
		# T_: GT
		writer = SummaryWriter()
		count = 0
		before_loss = 10000000
		for epoch in range(self.start, self.iter+1):
			for i, data in enumerate(self.train_loader):
				count+=1
				I_, GT_ = data
				# print 'GT:',GT_.shape
				loss_G, loss_D, loss_PL, loss_ML, loss_att, loss_MAP, MSE_loss= self.forward_process(I_,GT_)
				# print loss_G

				self.optim1.zero_grad()
				loss_G.backward(retain_graph=True)
				self.optim1.step()

				self.optim2.zero_grad()
				loss_D.backward()
				self.optim2.step()
				
				if count % 20==0:
					print('count: '+str(count)+' loss G: {:.4f}'.format(float(loss_G.data[0]))+' loss_D: {:.4f}'.format(float(loss_D.data[0]))+' loss_MSE: {:.4f}'.format(MSE_loss.data[0]))
					print('loss_PL:{:.4f}'.format(float(loss_PL.data[0]))+' loss_ML:{:.4f}'.format(float(loss_ML.data[0]))+' loss_Att:{:.4f}'.format(float(loss_att.data[0]))+' loss_MAP:{:.4f}'.format(float(loss_MAP.data[0])))
					writer.add_scalar('loss_G', float(loss_G.data[0]), count)
					writer.add_scalar('loss_D', float(loss_D.data[0]), count)
					
			step = 0
			for i, data in enumerate(self.valid_loader):
				I_, GT_ = data
				if i == 0:
					valid_loss_sum = self.forward_process(I_,GT_, is_train=False)
				else:
					valid_loss_sum += self.forward_process(I_,GT_, is_train=False)
				step+=1

			print('epoch_'+str(epoch)+'valid_loss:{} '.format(valid_loss_sum.data[0]/step)+'\n')
			writer.add_scalar('validation_loss', float(valid_loss_sum.data[0])/step, epoch)
			valid_loss_sum = float(valid_loss_sum.data[0])/step
			if before_loss/valid_loss_sum >1.01:
				before_loss = valid_loss_sum
				print("save model " + "!"*10)
				if not os.path.exists(self.out_path):
					os.system('mkdir -p {}'.format(self.out_path))
				w_name = 'G_epoch:{}_loss:{}.pth'.format(epoch,valid_loss_sum)
				save_path = os.path.join(self.out_path,w_name)
				torch.save(self.net_G.state_dict(), save_path)
				w_name = 'D_epoch:{}_loss:{}.pth'.format(epoch,valid_loss_sum)
				save_path = os.path.join(self.out_path,w_name)
				torch.save(self.net_D.state_dict(), save_path)
			valid_loss_sum = 0.
		
		writer.export_scalars_to_json("./attention_video_restoration.json")
		writer.close()
		return