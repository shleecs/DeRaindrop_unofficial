import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch.autograd import Variable
def get_heatmap(mask):
	lum_img = np.maximum(np.maximum(mask[:,:,0],mask[:,:,1],),mask[:,:,2])
	imgplot = plt.imshow(lum_img)
	imgplot.set_cmap('jet')
	plt.colorbar()
	plt.axis('off')
	pylab.show()
	return

def get_mask(dg_img,img):
	# downgraded image - image
	mask = np.fabs(dg_img-img)
	# threshold under 30
	mask[np.where(mask<(30.0/255.0))] = 0.0
	mask[np.where(mask>0.0)] = 1.0
	#avg? max?
	# mask = np.average(mask, axis=2)
	mask = np.max(mask, axis=2)
	mask = np.expand_dims(mask, axis=2)
	return mask

def torch_variable(x, is_train):
	if is_train:
		return Variable(torch.from_numpy(np.array(x).transpose((0,3,1,2))),requires_grad=True).cuda()
	else:
		return Variable(torch.from_numpy(np.array(x).transpose((0,3,1,2))),volatile=True).cuda()
