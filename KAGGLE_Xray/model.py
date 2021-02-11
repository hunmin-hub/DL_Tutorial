import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseModel(nn.Module):
    def __init__(self,image_size=224,in_channel=1,block_count=3,out_channel=1,k=12):
        super(DenseModel,self).__init__()
        self.image_size=image_size
        self.in_channel=in_channel
        self.block_count=block_count #Dense block loof
        self.out_channel=out_channel
        self.k=k #DenseNet 'K'
        # Nets
        self.start_net=self.start_layer()
        self.dense_nets=nn.ModuleList()
        self.conv_nets=nn.ModuleList()
        prev_channel=self.k*2
        for _ in range(block_count):
            self.dense_nets.append(self.dense_block(prev_channel))
            self.conv_nets.append(self.conv_layer(prev_channel*2))
        self.output_net=self.output_layer(prev_channel)
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                #nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def start_layer(self):
        return nn.Sequential(nn.Conv2d(in_channels=self.in_channel,
                                out_channels=self.k*2,
                                kernel_size=(3,3),
                                stride=(1,1),
                                padding=(1,1),
                                bias=False))
    
    def dense_block(self,in_channel):
        layers=[]
        layers.append(nn.BatchNorm2d(in_channel))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(in_channels=in_channel,
                                out_channels=in_channel*4,
                                kernel_size=(1,1),
                                stride=(1,1),
                                bias=False))
        layers.append(nn.BatchNorm2d(in_channel*4))
        layers.append(nn.Conv2d(in_channels=in_channel*4,
                                out_channels=in_channel,
                                kernel_size=(3,3),
                                stride=(1,1),
                                padding=(1,1),
                                bias=False))
        return nn.Sequential(*layers)
    
    def conv_layer(self,in_channel):
        layers=[]
        layers.append(nn.BatchNorm2d(in_channel))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(in_channels=in_channel,
                                out_channels=(in_channel//2),
                                kernel_size=(1,1),
                                stride=(1,1),
                                bias=False))
        layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        return nn.Sequential(*layers)

    def output_layer(self,in_channel):
        return nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                    out_channels=1,
                                    kernel_size=(1,1),
                                    stride=(1,1),
                                    bias=False))
    def forward(self,x):
        x=self.start_net(x)
        for idx in range(self.block_count):
            x=torch.cat([self.dense_nets[idx](x),x],dim=1)
            x=self.conv_nets[idx](x)
        x=self.output_net(x)
        x=x.view(-1)
        return x
        