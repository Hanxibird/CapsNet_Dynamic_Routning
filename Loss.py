import numpy as np
import torch
import torchvision
from torch.autograd import Variable


def to_categorical(y,num_classes):
    new_y=torch.eye(num_classes)[y.cpu().data.numpy(),]
    if(y.is_cuda):
        return new_y.cuda()

    return new_y

class MarginLoss(torch.nn.Module):
    def __init__(self,m_pos=0.9,m_neg=0.1,lamb=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos=m_pos
        self.m_neg=m_neg
        self.lamb=lamb

    def forward(self,score,y):
        y=Variable(to_categorical(y,10))

        Tc=y.float()
        loss_pos=torch.pow(torch.clamp(self.m_pos-score,min=0),2)
        loss_neg=torch.pow(torch.clamp(score-self.m_neg,min=0),2)
        loss=Tc*loss_pos+self.lamb*(1-Tc)*loss_neg
        loss=loss.sum(-1)
        return loss.mean()

class ReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self,x_reconstruction,x):
        loss = torch.pow(x - x_reconstruction, 2).sum(-1).sum(-1)
        return loss.mean()

class Capsule_Loss(torch.nn.Module):
    def __init__(self,reconstruction_loss_scale=0.0005):
        super(Capsule_Loss, self).__init__()
        self.marginloss=MarginLoss()
        self.reconstruction_loss=ReconstructionLoss()
        self.reconstruction_loss_scale=reconstruction_loss_scale

    def forward(self,x,y,x_reconstruction,y_pred):
        margin_loss=self.marginloss(y_pred.cuda(),y)
        reconstruction_loss=self.reconstruction_loss_scale *\
                            self.reconstruction_loss(x_reconstruction,x)

        loss = margin_loss + reconstruction_loss
        return loss, margin_loss, reconstruction_loss
