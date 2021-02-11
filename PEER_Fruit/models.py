import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    # EnsembleModel INFO
    # 1. Model_A : Normal CNN Model (Filter Size=3)
    # 2. Model_B : Normal CNN Model (Filter Size=5)
    # 3. Model_C : Custom ResNet Model (Filter Size=3)
    # 4. Model_D : Custom ResNet Model (Filter Size=5)
    # 5. Model_E : Custom DenseNet Model (Filter Size=3)
    def __init__(self,Model_A,Model_B,Model_C,Model_D,Model_E):
        super(EnsembleModel,self).__init__()
        self.Model_A=Model_A
        self.Model_B=Model_B
        self.Model_C=Model_C
        self.Model_D=Model_D
        self.Model_E=Model_E
        #self.init_params()
    
    def forward(self,x1,x2,x3,x4,x5):
        a=self.Model_A(x1)
        b=self.Model_B(x2)
        c=self.Model_C(x3)
        d=self.Model_D(x4)
        e=self.Model_E(x5)
        x=a+b+c+d+e
        return x
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ModelA(nn.Module):
    # Use Only Kernel_size=3,Padding=1
    # image_size = 224 * 224
    def __init__(self,image_size=224,in_channel=3,hidden_channels=[64,128,64,32],output_channel=6):
        super(ModelA,self).__init__()
        self.kernel_size=3
        self.image_size=image_size
        self.in_channel=in_channel
        self.hidden_channels=hidden_channels
        self.output_channel=output_channel
        layers=[]
        prev_channel=self.in_channel
        for current_channel in hidden_channels :
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            prev_channel=current_channel
        current_image_size=int(self.image_size/2**len(self.hidden_channels))
        layers.append(nn.Conv2d(in_channels=prev_channel,
                                out_channels=self.output_channel,
                                kernel_size=current_image_size,
                                stride=(1,1)))
        self.net=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.net(x)
        return F.log_softmax(x,dim=1)

class ModelB(nn.Module):
    # Use Only Kernel_size=5,Padding=2
    # image_size = 224 * 224
    def __init__(self,image_size=224,in_channel=3,hidden_channels=[64,128,64,32],output_channel=6):
        super(ModelB,self).__init__()
        self.kernel_size=5
        self.image_size=image_size
        self.in_channel=in_channel
        self.hidden_channels=hidden_channels
        self.output_channel=output_channel
        layers=[]
        prev_channel=self.in_channel
        for current_channel in hidden_channels :
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            prev_channel=current_channel
        current_image_size=int(self.image_size/2**len(self.hidden_channels))
        layers.append(nn.Conv2d(in_channels=prev_channel,
                                out_channels=output_channel,
                                kernel_size=current_image_size,
                                stride=(1,1)))
        self.net=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.net(x)
        return F.log_softmax(x,dim=1)

class ModelC(nn.Module):
    # Use Only Kernel_size=3,Padding=1
    # f(x)+x , look like ResNet
    # image_size = 224 * 224
    def __init__(self,image_size=224,in_channel=3,block_channels=[64,128,256],output_channel=6):
        super(ModelC,self).__init__()
        self.kernel_size=3
        self.image_size=image_size
        self.in_channel=in_channel
        self.block_channels=block_channels
        self.output_channel=output_channel
        #Layers Setting
        self.MaxPool_net=nn.Sequential(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.start_net=self.start_layer()

        self.block_net=nn.ModuleList()
        self.change_channel_net=nn.ModuleList()
        for idx,input_channel in enumerate(self.block_channels):
            self.block_net.append(self.block_layer(idx))
            if idx+1<len(self.block_channels) :
                self.change_channel_net.append(self.change_channel(input_channel,self.block_channels[idx+1]))
            else :
                self.change_channel_net.append(self.change_channel(input_channel,64))

        current_image_size=int(self.image_size/2**(len(self.block_channels)+1))
        self.output_net=nn.Sequential(nn.Conv2d(in_channels=64,
                                out_channels=output_channel,
                                kernel_size=current_image_size,
                                stride=(1,1)))

    def start_layer(self):
        layers=[]
        layers.append(nn.Conv2d(in_channels=self.in_channel,
                                    out_channels=self.block_channels[0],
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
        layers.append(nn.BatchNorm2d(self.block_channels[0]))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        return nn.Sequential(*layers)

    def change_channel(self,in_channel,out_channel):
        return nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=(1,1)))

    def block_layer(self,block_depth):
        self.block_depth=block_depth
        layers=[]
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=self.block_channels[self.block_depth],
                                        out_channels=self.block_channels[self.block_depth],
                                        kernel_size=self.kernel_size,
                                        stride=(1,1),
                                        padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(self.block_channels[self.block_depth]))
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.start_net(x)
        for idx,current_block in enumerate(self.block_net):
            identity=x
            x=current_block(x)+identity
            x=self.MaxPool_net(x)
            x=self.change_channel_net[idx](x)
        x=self.output_net(x)
        return F.log_softmax(x,dim=1)

class ModelD(nn.Module):
    # Use Only Kernel_size=5,Padding=2
    # f(x)+x , look like ResNet
    # image_size = 224 * 224
    def __init__(self,image_size=224,in_channel=3,block_channels=[64,128,256],output_channel=6):
        super(ModelD,self).__init__()
        self.kernel_size=3
        self.image_size=image_size
        self.in_channel=in_channel
        self.block_channels=block_channels
        self.output_channel=output_channel
        #Layers Setting
        self.MaxPool_net=nn.Sequential(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.start_net=self.start_layer()

        self.block_net=nn.ModuleList()
        self.change_channel_net=nn.ModuleList()
        for idx,input_channel in enumerate(self.block_channels):
            self.block_net.append(self.block_layer(idx))
            if idx+1<len(self.block_channels) :
                self.change_channel_net.append(self.change_channel(input_channel,self.block_channels[idx+1]))
            else :
                self.change_channel_net.append(self.change_channel(input_channel,64))

        current_image_size=int(self.image_size/2**(len(self.block_channels)+1))
        self.output_net=nn.Sequential(nn.Conv2d(in_channels=64,
                                out_channels=output_channel,
                                kernel_size=current_image_size,
                                stride=(1,1)))

    def start_layer(self):
        layers=[]
        layers.append(nn.Conv2d(in_channels=self.in_channel,
                                    out_channels=self.block_channels[0],
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
        layers.append(nn.BatchNorm2d(self.block_channels[0]))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        return nn.Sequential(*layers)

    def change_channel(self,in_channel,out_channel):
        return nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=(1,1)))

    def block_layer(self,block_depth):
        self.block_depth=block_depth
        layers=[]
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=self.block_channels[self.block_depth],
                                        out_channels=self.block_channels[self.block_depth],
                                        kernel_size=self.kernel_size,
                                        stride=(1,1),
                                        padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(self.block_channels[self.block_depth]))
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.start_net(x)
        for idx,current_block in enumerate(self.block_net):
            identity=x
            x=current_block(x)+identity
            x=self.MaxPool_net(x)
            x=self.change_channel_net[idx](x)
        x=self.output_net(x)
        return F.log_softmax(x,dim=1)

class ModelE(nn.Module):
    # like DenseNet
    def __init__(self,image_size=224,in_channel=3,hidden_channels=[64,128],out_channel=6):
        super(ModelE,self).__init__()
        self.kernel_size=3
        self.image_size=image_size
        self.in_channel=in_channel
        self.hidden_channels=hidden_channels
        self.out_channel=out_channel
        self.MaxPool_net=nn.Sequential(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))

        self.block_net=nn.ModuleList()
        prev_channel=self.in_channel
        for current_channel in hidden_channels:
            layers=[]
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            prev_channel=current_channel+prev_channel
            self.block_net.append(nn.Sequential(*layers))
        
        current_image_size=int(self.image_size/2**(len(self.hidden_channels)))
        self.output_net=nn.Sequential(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=self.out_channel,
                                    kernel_size=current_image_size,
                                    stride=(1,1)))
        
    def forward(self,x):
        for current_dense_net in self.block_net:
            identity=x
            x=torch.cat((current_dense_net(x),identity),dim=1)
            x=self.MaxPool_net(x)
        x=self.output_net(x)
        return F.log_softmax(x,dim=1)

