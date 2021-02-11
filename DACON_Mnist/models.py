import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self,Model_A,Model_B,Model_C):
        super(EnsembleModel,self).__init__()
        self.Model_A=Model_A
        self.Model_B=Model_B
        self.Model_C=Model_C
        self.init_params()

    def forward(self,x1,x2,x3):
        x1=self.Model_A(x1)
        x2=self.Model_B(x2)
        x3=self.Model_C(x3)
        x=x1+x2+x3
        return F.log_softmax(x,dim=1)

    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LinearModel(nn.Module): 
    # only linear model
    def __init__(self,in_channel=28*28,linear_channel=[256,64,32,10]):
        super(LinearModel,self).__init__()
        self.in_channel=in_channel
        self.linear_channel=linear_channel
        layers=[]
        prev_channel=in_channel
        for current_channel in linear_channel :
            layers.append(nn.Linear(in_features=prev_channel,
                                    out_features=current_channel,bias=True))
            layers.append(nn.BatchNorm1d(current_channel))
            layers.append(nn.ReLU(inplace=True))
            prev_channel=current_channel
        self.net=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.net(x)
        return F.log_softmax(x,dim=1)

class CnnModel(nn.Module):
    # Normal CNN
    def __init__(self,image_size=[28,28],in_channel=1,hidden_channel=[32,64,32,10]):
        super(CnnModel,self).__init__()
        self.image_size=image_size
        self.in_channel=in_channel
        self.hidden_channel=hidden_channel
        self.kernel_size=3
        layers=[]
        prev_channel=in_channel
        # 28 -> 14 -> 7 -> 3 -> 1
        for current_channel in hidden_channel :
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            prev_channel=current_channel
        self.net=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.net(x)
        x=x.reshape(-1,10)
        return F.log_softmax(x,dim=1)
        
class CustomModel(nn.Module):
    # Custom Model (like Resnet)
    def __init__(self,image_size=[28,28],in_channel=1,block1_channel=[64,32,64],block2_channel=[64,32,64],out_channel=10):
        super(CustomModel,self).__init__()
        self.image_size=image_size
        self.kernel_size=3
        self.in_channel=in_channel
        self.block1_channel=block1_channel
        self.block2_channel=block2_channel
        self.out_channel=out_channel
        self.MaxPool_net=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        # Start layers (imagesize=28->14)
        start_layers=[]
        start_layers.append(nn.Conv2d(in_channels=self.in_channel,
                                    out_channels=self.block1_channel[0],
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
        start_layers.append(nn.BatchNorm2d(self.block1_channel[0]))
        start_layers.append(nn.ReLU(True))
        start_layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))

        self.start_net=nn.Sequential(*start_layers)

        #Block1 layers ( 14->7)
        block1_layers=[]
        prev_channel=self.block1_channel[0]
        for current_channel in block1_channel:
            block1_layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            block1_layers.append(nn.BatchNorm2d(current_channel))
            block1_layers.append(nn.ReLU(True))
            prev_channel=current_channel
        
        self.block1_net=nn.Sequential(*block1_layers)

        #Block2 layers (7->3)
        block2_layers=[]
        for current_channel in block2_channel:
            block2_layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            block2_layers.append(nn.BatchNorm2d(current_channel))
            block2_layers.append(nn.ReLU(True))
            prev_channel=current_channel
        
        self.block2_net=nn.Sequential(*block2_layers)

        #output layers
        output_layers=[]
        output_layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=out_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1)))
        output_layers.append(nn.BatchNorm2d(out_channel))
        output_layers.append(nn.ReLU(True))

        self.output_net=nn.Sequential(*output_layers)

    def forward(self,x):
        x=self.start_net(x) # 14
        identity=x
        x=self.block1_net(x)+identity
        x=self.MaxPool_net(x) # 7
        identity=x
        x=self.block2_net(x)+identity
        x=self.MaxPool_net(x) # 3
        x=self.output_net(x)
        x=x.view(-1,10)
        return F.log_softmax(x,dim=1)