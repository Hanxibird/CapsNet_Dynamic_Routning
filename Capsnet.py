import numpy as np
import torch
from torch.autograd import Variable


class Conv1(torch.nn.Module):
    def __init__(self,input_channels,output_channels=256,kernel_size=9):
        super(Conv1,self).__init__()
        self.conv=torch.nn.Conv2d(input_channels,output_channels,kernel_size)
        self.activation=torch.nn.ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.activation(x)
        return x

class PrimaryCapsules(torch.nn.Module):
    def __init__(self,input_shape=(256,20,20),capsule_dim=8,output_channels=32,kernel_size=9,stride=2):
        super(PrimaryCapsules,self).__init__()
        self.input_shape=input_shape
        self.capsule_dim=capsule_dim
        self.output_channels=output_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.input_channels=input_shape[0]

        self.conv=torch.nn.Conv2d(
            self.input_channels,
            self.output_channels*self.capsule_dim,
            self.kernel_size,
            self.stride
        )
    def forward(self,x):
        x=self.conv(x)
        x=x.permute(0,2,3,1).contiguous()
        x=x.view(-1,x.size()[1],x.size()[2],self.output_channels,self.capsule_dim)
        return x




class Routing(torch.nn.Module):
    def __init__(self,caps_dim_before=8,caps_dim_after=16,n_capsules_before=(6*6*32),n_capsules_after=10):
        super(Routing,self).__init__()
        self.n_capsules_before = n_capsules_before
        self.n_capsules_after = n_capsules_after
        self.caps_dim_before = caps_dim_before
        self.caps_dim_after = caps_dim_after
        n_in=self.n_capsules_before*self.caps_dim_before
        variance=2/n_in
        std=np.sqrt(variance)
        self.W=torch.nn.Parameter(
            torch.randn(
                self.n_capsules_before,
                self.n_capsules_after,
                self.caps_dim_after,
                self.caps_dim_before)*std,
            requires_grad=True
        )

    def squash(s):
        s_norm=torch.norm(s,p=2,dim=-1,keepdim=True)
        s_norm2=torch.pow(s_norm,2)
        v=(s_norm2/(1+s_norm2))*(s/s_norm)
        return v

    def affine(self,x):
        x=self.W @ x.unsqueeze(2).expand(-1,-1,10,-1).unsqueeze(-1)
        return x.squeeze()


    def softmax(x, dim=-1):
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim, keepdim=True)

    def routing(self,u,r,l):
        b = Variable(torch.zeros(u.size()[0], l[0], l[1]), requires_grad=False).cuda()
        for i in range(r):
            c=Routing.softmax(b)
            s=(c.unsqueeze(-1).expand(-1,-1,-1,u.size()[-1])*u).sum(1)
            v=Routing.squash(s)
            b+=(u*v.unsqueeze(1).expand(-1,l[0],-1,-1)).sum(-1)
        return v

    def forward(self,x,n_routing_iter):
        x=x.view((-1,self.n_capsules_before,self.caps_dim_before))
        x=self.affine(x)
        x = self.routing(x, n_routing_iter, (self.n_capsules_before, self.n_capsules_after))
        return x

class Norm(torch.nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self,x):
        x=torch.norm(x,p=2,dim=-1)
        return x

class Decoder(torch.nn.Module):
    def __init__(self,in_features,out_features,output_size=(1,28,28)):
        super(Decoder, self).__init__()
        self.decoder=self.assemble_decoder(in_features,out_features)
        self.output_size=output_size

    def assemble_decoder(self,in_features,out_features):
        hidden_layer_features=[512,1024]
        return torch.nn.Sequential(
            torch.nn.Linear(in_features,hidden_layer_features[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_features[0], hidden_layer_features[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_features[1],out_features),
            torch.nn.Sigmoid()
        )

    def forward(self,x,y):
        x = x[np.arange(0, x.size()[0]), y.cpu().data.numpy(), :].cuda()
        x = self.decoder(x)
        x = x.view(*((-1,) + self.output_size))
        return x

class CapsNet(torch.nn.Module):
    def __init__(self,input_shape=(1,28,28),n_routing_iter=3,use_reconstruction=True):
        super(CapsNet, self).__init__()
        assert len(input_shape)==3

        self.input_shape=input_shape
        self.n_routing_iter=n_routing_iter
        self.use_reconstruction=use_reconstruction

        self.conv1=Conv1(input_shape[0],256,9)
        self.primary_capsules=PrimaryCapsules(
            input_shape=(256,20,20),
            capsule_dim=8,
            output_channels=32,
            kernel_size=9,
            stride=2
        )
        self.routing=Routing(
            caps_dim_before=8,
            caps_dim_after=16,
            n_capsules_before=1152,
            n_capsules_after=10
        )

        self.norm=Norm()

        if(self.use_reconstruction):
            self.decoder=Decoder(16,int(np.prod(input_shape)))

    def n_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def forward(self,x,y=None):
        conv1=self.conv1(x)
        primary_capsule=self.primary_capsules(conv1)
        digit_capsules=self.routing(primary_capsule,self.n_routing_iter)
        scores=self.norm(digit_capsules)

        if(self.use_reconstruction and y is not None):
            reconstruction=self.decoder(digit_capsules,y).view((-1,)+self.input_shape)
            return scores,reconstruction

        return scores






            

