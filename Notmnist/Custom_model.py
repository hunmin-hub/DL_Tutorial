import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# CONFIG
torch.manual_seed(53)
BATCH_SIZE=256
EPOCHS=10
# DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def get_data() :
    print("Start Data setting...")
    data_transforms=transforms.Compose([transforms.ToTensor()])
    train_data=datasets.ImageFolder('./dataset/train',transform=data_transforms)
    test_data=datasets.ImageFolder('./dataset/test',transform=data_transforms)
    # data to batch type(tensor)
    train_iter=torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
    test_iter=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
    print("Data setting Done.\n")
    return train_iter,test_iter

class CNN(nn.Module):
    def __init__(self,name='cnn',imageSize=[28,28],
                                input_channel=3,hidden_channel=32,block_count=3,output_channel=10) :
        super(CNN,self).__init__()
        # setting
        # general : Convolution - Batch Normalization - Activation - Dropout - Pooling
        self.imageSize=imageSize
        self.input_channel=input_channel
        self.hidden_channel=hidden_channel
        self.block_count=block_count
        self.output_channel=output_channel
        self.kernel_size=3 # kernel size : 3*3
        self.relu=nn.ReLU(inplace=True)
        # layers setting
        # 1. Start_net -> 2. loop 3 : (Block_net) -> 3. Output_net
        prev_channel=self.hidden_channel
        self.MaxPool_net=nn.Sequential(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.start_net=self.start_layer(prev_channel) ## start_net
        self.block_layers=nn.ModuleList()
        self.change_channel=nn.ModuleList()
        current_channel=self.hidden_channel
        # Make module list
        for idx in range(3) :
            self.change_channel.append(nn.Sequential(nn.Conv2d(in_channels=prev_channel,
                                                out_channels=current_channel,
                                                kernel_size=(1,1),
                                                stride=(1,1))))
            self.block_layers.append(self.block_layer(prev_channel,current_channel))
            prev_channel=current_channel
            current_channel=current_channel*2

        self.output_net=nn.Sequential(nn.Conv2d(in_channels=prev_channel,
                                                out_channels=self.output_channel,
                                                kernel_size=(1,1),
                                                stride=(1,1)))
        self.init_params()

    def start_layer(self,hidden_channel) :
        layers=[]
        layers.append(nn.Conv2d(in_channels=3,
                                    out_channels=hidden_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=(self.kernel_size-1)//2))
        layers.append(nn.BatchNorm2d(hidden_channel))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        return nn.Sequential(*layers)

    def block_layer(self,prev_channel,current_channel) :
        layers=[]
        layers.append(nn.Conv2d(in_channels=prev_channel,
                                out_channels=current_channel,
                                kernel_size=self.kernel_size,
                                stride=(1,1),
                                padding=(self.kernel_size-1)//2))
        layers.append(nn.BatchNorm2d(current_channel))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(in_channels=current_channel,
                                out_channels=current_channel,
                                kernel_size=self.kernel_size,
                                stride=(1,1),
                                padding=(self.kernel_size-1)//2))
        layers.append(nn.BatchNorm2d(current_channel))
        layers.append(nn.ReLU(True))                           
        return nn.Sequential(*layers)
        
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.start_net(x)
        for idx in range(3) :
            identity=self.change_channel[idx](x)
            x=self.block_layers[idx](x)+identity
            x=self.relu(x)
            x=self.MaxPool_net(x)
        return self.output_net(x)
    
def count_parameters(Model) :
    total_params = 0
    for param_name, param in Model.named_parameters():
        if param.requires_grad:
            print(f'[{param_name}] Layers Parameters : [{len(param.reshape(-1))}] shape : [{param.size()}]')
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")

def func_eval(Model,data_iter):
    with torch.no_grad():
        Model.eval()
        n_total, n_correct = 0,0
        print("Testing......\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            y_target=batch_out.to(device)
            model_pred=Model(batch_in.view(-1,3,28,28).to(device))
            model_pred=model_pred.reshape(-1,10)
            _, y_pred=torch.max(model_pred.data,1)
            n_correct+=(y_pred==y_target).sum().item()
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        Model.train()
    print("Testing Done.\n")
    return val_accr

def model_train(Model,train_iter,test_iter,optm,loss):
    Model.init_params()
    Model.train()
    print("Start Training....\n")
    print_every=1
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)):
            y_pred=Model.forward(batch_in.view(-1,3,28,28).to(device))
            y_pred=y_pred.reshape(-1,10)
            loss_out=loss(y_pred,batch_out.to(device))
            # params update
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum+=loss_out
        loss_val_avg=loss_val_sum/len(train_iter)
        if epoch==EPOCHS-1 : # first & last epoch -> test
            train_accr=func_eval(Model,train_iter)
            test_accr=func_eval(Model,test_iter)
            print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]"%(epoch,loss_val_avg,train_accr,test_accr))
    print("Training Done.")
    return

def image_show(image, ax=None,title=None,normalize=True) :
    if ax is None:
        flag, ax = plt.subplots()
    image=image.numpy().transpose((1,2,0))

    if normalize:
        # Normalize Image -> Origin Image
        mean=np.array([0,0,0])
        std=np.array([0,0,0])
        image=std*image+mean
        image=np.clip(image,0,1)
    
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(title)
    return ax

def test_show(Model,test_iter) :
    images, labels = next(iter(test_iter))
    # I want to show image (sample=16)
    n_sample=16
    test_x=images[:n_sample]
    test_y=labels[:n_sample]

    # get Model pred by Sample image
    with torch.no_grad():
        Model.eval()
        y_pred = Model.forward(test_x.view(-1,3,28,28).type(torch.float).to(device))
        y_pred = y_pred.reshape(-1,10)
        Model.train()
    y_pred=y_pred.argmax(axis=1)

    # Image show
    ordered=['A','B','C','D','E','F','G','H','I','J']
    plt.figure(figsize=(20,20))
    for idx in range(n_sample):
        ax=plt.subplot(4,4,idx+1)
        title=f"Predict:{ordered[y_pred[idx]]}, Label:{ordered[test_y[idx]]}"
        image_show(test_x[idx],ax,title,normalize=False)
    plt.show()

def main() :
    train_iter, test_iter = get_data()
    # ---------------------- Model Setting ----------------------------#
    # Image Size : 28*28 / input channel : 3 (RGB) / 
    Model=CNN(name='cnn',imageSize=[28,28],input_channel=3,hidden_channel=32,block_count=3,output_channel=10).to(device)
    loss=nn.CrossEntropyLoss() # with SOFTMax
    optm=optim.Adam(Model.parameters(),lr=1e-3)
    count_parameters(Model)
    model_train(Model,train_iter,test_iter,optm,loss)
    test_show(Model,test_iter) # 16 image test & show

if __name__ == "__main__" :
    main()