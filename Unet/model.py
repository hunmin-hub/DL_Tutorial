import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.normal_pooling=nn.MaxPool2d(kernel_size=2) # Encoder Pooling Net
        # Encoder
        self.encoder_1=self.encoder_block(1,64)
        self.encoder_2=self.encoder_block(64,128)
        self.encoder_3=self.encoder_block(128,256)
        self.encoder_4=self.encoder_block(256,512)

        self.bottleneck=self.decoder_block(512,1024,512) # Bottleneck

        # Decoder
        self.un_pooling_4=self.un_pool(512,512)
        self.decoder_4=self.decoder_block(1024,512,256)
        self.un_pooling_3=self.un_pool(256,256)
        self.decoder_3=self.decoder_block(512,256,128)
        self.un_pooling_2=self.un_pool(128,128)
        self.decoder_2=self.decoder_block(256,128,64)
        self.un_pooling_1=self.un_pool(64,64)
        self.decoder_1=self.decoder_block(128,64,64)

        self.output_net=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1),padding=0))

    def un_pool(self,in_channel,out_channel):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,
                                                out_channels=out_channel,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0))

    def encoder_block(self,in_channel,out_channel):
        layers=[]
        prev_channel=in_channel
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=out_channel,
                                    kernel_size=(3,3),
                                    stride=(1,1),
                                    padding=(1,1)))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(True))
            prev_channel=out_channel
        return nn.Sequential(*layers)

    def decoder_block(self,in_channel,hidden_channel,out_channel):
        layers=[]
        layers.append(nn.Conv2d(in_channels=in_channel,
                                    out_channels=hidden_channel,
                                    kernel_size=(3,3),
                                    stride=(1,1),
                                    padding=(1,1)))
        layers.append(nn.BatchNorm2d(hidden_channel))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(in_channels=hidden_channel,
                                    out_channels=out_channel,
                                    kernel_size=(3,3),
                                    stride=(1,1),
                                    padding=(1,1)))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.encoder_1(x)
        encoder_x1=x
        x=self.normal_pooling(x)
        x=self.encoder_2(x)
        encoder_x2=x
        x=self.normal_pooling(x)
        x=self.encoder_3(x)
        encoder_x3=x
        x=self.normal_pooling(x)
        x=self.encoder_4(x)
        encoder_x4=x
        x=self.normal_pooling(x)
        x=self.bottleneck(x)
        x=self.un_pooling_4(x)
        x=self.decoder_4(torch.cat([x,encoder_x4],dim=1))
        x=self.un_pooling_3(x)
        x=self.decoder_3(torch.cat([x,encoder_x3],dim=1))
        x=self.un_pooling_2(x)
        x=self.decoder_2(torch.cat([x,encoder_x2],dim=1))
        x=self.un_pooling_1(x)
        x=self.decoder_1(torch.cat([x,encoder_x1],dim=1))
        x=self.output_net(x)
        return x
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)