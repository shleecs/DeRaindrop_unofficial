import torch
import torch.nn as nn
import torchvision
import cv2
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import numpy as np

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        # self.loss = nn.MSELoss().cuda()
        self.loss = nn.BCELoss().cuda()
    def convert_tensor(self, input, is_real):
        if is_real:
            return Variable(torch.FloatTensor(input.size()).fill_(self.real_label)).cuda()
        else:
            return Variable(torch.FloatTensor(input.size()).fill_(self.fake_label)).cuda() 
    def __call__(self, input, is_real):
        return self.loss(input, self.convert_tensor(input,is_real).cuda())

class AttentionLoss(nn.Module):
    def __init__(self, theta=0.8, iteration=4):
        super(AttentionLoss, self).__init__()
        self.theta = theta
        self.iteration = iteration
        self.loss = nn.MSELoss().cuda()

    def __call__(self, A_, M_):
        loss_ATT = None
        for i in range(1, self.iteration+1):
            if i == 1:
                loss_ATT = pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
            else:
                loss_ATT += pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
        return loss_ATT

# VGG16 pretrained on Imagenet
def trainable(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.model = (vgg16(pretrained = True).cuda())
        trainable(self.model, False)

        self.loss = nn.MSELoss().cuda()
        self.vgg_layers = self.model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }
    def get_layer_output(self,x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    def __call__(self, O_, T_):
        o = self.get_layer_output(O_)
        t = self.get_layer_output(T_)
        loss_PL = None
        for i in range(len(t)):
            if i == 0:
                loss_PL = self.loss(o[i],t[i]) / float(len(t))
            else:
                loss_PL += self.loss(o[i],t[i]) / float(len(t))
        return loss_PL
        
class MultiscaleLoss(nn.Module):
    def __init__(self, ld=[0.6,0.8,1.0],batch=1):
        super(MultiscaleLoss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        self.ld = ld
        self.batch=batch
    def __call__(self, S_, gt):
        #1,128,256,3
        T_ = []
        # print S_[0].shape[0]
        for i in range(S_[0].shape[0]):
            temp = []
            x = (np.array(gt[i])*255.).astype(np.uint8)
            # print (x.shape, x.dtype)
            t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(t)
            t = cv2.resize(x, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(t)
            x = np.expand_dims((x/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(x)
            T_.append(temp)
        temp_T = []
        for i in range(len(self.ld)):
            # if self.batch == 1:
            #     temp_T.append(Variable(torch.from_numpy(T_[0][i])).cuda())
            # else:
            for j in range((S_[0].shape[0])):
                if j == 0:
                    x = T_[j][i]
                else:
                    x = np.concatenate((x, T_[j][i]), axis=0)
            temp_T.append(Variable(torch.from_numpy(x)).cuda())
        T_ = temp_T
        loss_ML = None
        for i in range(len(self.ld)):
            if i == 0: 
                loss_ML = self.ld[i] * self.loss(S_[i], T_[i])
            else:
                loss_ML += self.ld[i] * self.loss(S_[i], T_[i])
        
        return loss_ML/float(S_[0].shape[0])

class MAPLoss(nn.Module):
    def __init__(self, gamma=0.05):
        super(MAPLoss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        self.gamma = gamma

    # D_map_O, D_map_R
    def __call__(self, D_O, D_R, A_N):
        Z = Variable(torch.zeros(D_R.shape)).cuda()
        D_A = self.loss(D_O,A_N)
        D_Z = self.loss(D_R,Z)
        return self.gamma * (D_A + D_Z)
