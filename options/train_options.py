import argparse
import torch
import os

class TrainOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False
		self.opt = None
	def initialize(self):
		self.parser.add_argument('--gpu', type=str, default='0', help='gpu: e.g. 0  0,1 1,2.')
		self.parser.add_argument('--checkpoint_ext', type=str, default='pkl', help='checkpoint extension')
		self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='path to save model')
		self.parser.add_argument('--load', type=int, default=-1, help='epoch number which you want to load. use -1 for latest')
		self.parser.add_argument('--train_dataset', type=str, default='./dataset/train', help='path to training dataset')
		self.parser.add_argument('--eval_dataset', type=str, default='./dataset/test_a', help='path to evaluation dataset')
		self.parser.add_argument('--test_dataset', type=str, default='./dataset/test_b', help='path to test dataset')
		self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
		self.parser.add_argument('--iter', type=int, default=200, help='number of iterations')
		self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		gpu = list(map(int, self.opt.gpu.split(',')))
		self.opt.gpu = gpu
		if len(gpu) > 0:
			torch.cuda.set_device(self.opt.gpu[0])
		if self.opt.load < 0:
			files = os.listdir(self.opt.checkpoint_dir)
			cps = []
			for f in files:
				ext = os.path.splitext(f)[-1]
				if ext[1:] == self.opt.checkpoint_ext:
					e_ = f.split('_')[0]
					cps.append(int(e_[1:]))
				cps = sorted(cps)
				if len(cps) > 0:
					self.opt.load = int(cps[-1])
				else:
					self.opt.load = 1
		return self.opt