import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorModel(nn.Module):
    def __init__(self,z_size,in_channel=1):
        super(GeneratorModel,self).__init__()
        self.z_size=z_size
        self.in_channel=in_channel
        # Layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.z_size, 32, 3, 1, 0, bias=False), # 1 -> 3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False), # 3 -> 6
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 8, 4, 2, 0, bias=False), # 6 -> 14
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(8, 1, 4, 2, 1, bias=False), # 14 -> 28
            nn.Tanh()
        )

    def forward(self,x):
        x=self.net(x)
        return x.view(-1,1,28,28)
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight)
                #nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
