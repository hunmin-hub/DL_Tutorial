import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorModel(nn.Module):
    def __init__(self,in_channel=1,hidden_channels=[8,16,32]):
        super(DiscriminatorModel,self).__init__()
        self.in_channel=in_channel
        self.hidden_channels=hidden_channels
        prev_channel=self.in_channel
        layers=[]
        for current_channel in self.hidden_channels:
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.LeakyReLU(0.2,False))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            prev_channel=current_channel
        self.net=nn.Sequential(*layers)
        current_image_size=int(28//2**len(self.hidden_channels))
        self.out=nn.Sequential(nn.Conv2d(in_channels=prev_channel,
                                out_channels=1,
                                kernel_size=current_image_size,
                                stride=1,
                                bias=False))
  
    def forward(self,x):
        x=self.net(x)
        x=self.out(x)
        return x.view(-1)
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                #nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        