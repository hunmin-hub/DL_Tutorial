import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
#Config
torch.manual_seed(53)
BATCH_SIZE=128
EPOCHS=20
LEARNING_RATE=1e-3
checkpoint_dir="./weights"
#Device
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

class CNN(nn.Module):
    # Made by hunmin-hub
    def __init__(self,name='cnn',imageSize=[28,28],input_channel=1,start_channels=[32,64],block_channels=[128,256],block_loof=2,out_channel=10):
        super(CNN,self).__init__()
        self.kernel_size=3
        self.imageSize=imageSize
        self.input_channel=input_channel
        self.start_channels=start_channels
        self.block_channels=block_channels
        self.out_channel=out_channel
        self.block_loof=block_loof # block_loof count
        self.block_inchannel=start_channels[-1] # block input_channel = start_layers out channel
        #NET
        self.relu=nn.ReLU(inplace=True)
        self.start_net=self.start_layer()
        #Block Net
        self.block_net=self.make_blocklayer()
        #Output Net
        self.output_net=self.output_layer()
    
    def start_layer(self): # 1-> 32 -> 64 / Size : 28 -> 14 -> 7
        layers=[]
        prev_channel=self.input_channel
        for current_channel in self.start_channels :
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            #layers.append(nn.Dropout2d(p=0.5))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            prev_channel=current_channel
        return nn.Sequential(*layers)

    def make_blocklayer(self):
        layers=[]
        for _ in range(self.block_loof):
            layers.append(BlockNet(in_channel=self.block_inchannel,mid_channel=self.block_channels,out_channel=self.block_channels[0]))
            #layers.append(nn.Dropout2d(p=0.5))
            layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            self.block_inchannel=self.block_channels[0]
            for idx in range(len(self.block_channels)):
                self.block_channels[idx]*=2
        return nn.Sequential(*layers)
    
    def output_layer(self):
        layers=[]
        layers.append(nn.Conv2d(in_channels=self.block_inchannel,
                                out_channels=self.out_channel,
                                kernel_size=(1,1),
                                stride=(1,1)))
        return nn.Sequential(*layers)

    def forward(self,x) :
        x=self.start_net(x)
        x=self.block_net(x)
        x=self.output_net(x)
        return F.log_softmax(x,dim=1)

    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BlockNet(nn.Module):
    def __init__(self,in_channel=64,mid_channel=[128,256],out_channel=128):
        super(BlockNet,self).__init__()
        self.in_channel=in_channel
        self.mid_channel=mid_channel
        self.out_channel=out_channel
        self.kernel_size=3
        # net
        self.relu=nn.ReLU(inplace=True)
        self.channel_change_net=self.channel_change_layer()
        self.block_net=self.Block_layer()

    def channel_change_layer(self):
        layers=[]
        layers.append(nn.Conv2d(in_channels=self.in_channel,
                                out_channels=self.out_channel,
                                kernel_size=(1,1),
                                stride=(1,1)))
        layers.append(nn.BatchNorm2d(self.out_channel))
        return nn.Sequential(*layers)

    def Block_layer(self):
        layers=[]
        prev_channel=self.in_channel
        for current_channel in self.mid_channel :
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=current_channel,
                                    kernel_size=self.kernel_size,
                                    stride=(1,1),
                                    padding=((self.kernel_size-1)//2)))
            layers.append(nn.BatchNorm2d(current_channel))
            layers.append(nn.ReLU(True))
            prev_channel=current_channel
        layers.append(nn.Conv2d(in_channels=prev_channel,
                                out_channels=self.out_channel,
                                kernel_size=(1,1),
                                stride=(1,1)))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        identity=self.channel_change_net(x)
        x=self.block_net(x)
        x+=identity
        x=self.relu(x)
        return x

        
class DatasetFashion(data.Dataset):
    def __init__(self,file_path,test_mode=False,train=True,transform=None):
        data=pd.read_csv(file_path,index_col='index')
        self.test_mode=test_mode
        self.train=train
        self.dataframe=list(data.values)
        self.transform=transform
        # train, validation
        image=[]
        label=[]
        if test_mode==False :
            #Train mode get data -> train data (train,vali)
            train_len=int(len(self.dataframe)*0.8)
            if self.train : # train
                for idx in range(0,train_len) :
                    label.append(self.dataframe[idx][0])
                    image.append(self.dataframe[idx][1:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
            else : # validation
                for idx in range(train_len,len(self.dataframe)) :
                    label.append(self.dataframe[idx][0])
                    image.append(self.dataframe[idx][1:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
        else :
            # Last Test Mode -> All train (train+vali) -> test result (csv)
            # All train
            if self.train :
                for line in self.dataframe :
                    label.append(line[0])
                    image.append(line[1:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
            else :
                for line in self.dataframe :
                    label.append(0)
                    image.append(line[0:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        label=self.labels[index]
        image=self.images[index]
        if self.transform!=None:
            image=self.transform(image)
        return image,label

def count_parameters(Model) :
    total_params = 0
    for param_name, param in Model.named_parameters():
        if param.requires_grad:
            print(f'[{param_name}] Layers Parameters : [{len(param.reshape(-1))}] shape : [{param.size()}]')
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")

## Last Test to csv
def last_test(Model,test_iter):
    pred_label=[]
    with torch.no_grad():
        Model.eval()
        print("Test Data Testing.....\n")
        for batch_in, batch_out in tqdm(iter(test_iter)):
            model_pred=Model(batch_in.view(-1,1,28,28).to(device))
            model_pred=model_pred.reshape(-1,10)
            _, y_pred=torch.max(model_pred.data,1) # My Model pred
            pred_label.extend(y_pred.tolist())
    print("Testing done.\n")
    print("Result to submission.csv")
    submission = pd.read_csv('./datasets/sample_submission.csv', encoding = 'utf-8')
    submission['label'] = pred_label
    submission.to_csv('./datasets/fashion_submission.csv', index = False)
    return

def func_eval(Model,data_iter):
    with torch.no_grad():
        Model.eval()
        n_total, n_correct = 0,0
        print("(Train or Validation) Data Testing....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            y_target=batch_out.to(device)
            model_pred=Model(batch_in.view(-1,1,28,28).to(device))
            model_pred=model_pred.reshape(-1,10)
            _, y_pred=torch.max(model_pred.data,1)
            n_correct+=(y_pred==y_target).sum().item()
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        Model.train()
    print("Testing Done.\n")
    return val_accr

# ----------------------- Model Train ------------------------------------ #
def model_train(Model,train_iter,vali_iter,optm,loss,best_accr):
    if best_accr==0 :
        Model.init_params()
    else :
        Model.eval()
    Model.train()
    print("Start Training....\n")
    print_every=10
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)):
            y_pred=Model.forward(batch_in.view(-1,1,28,28).to(device))
            y_pred=y_pred.reshape(-1,10)
            loss_out=loss(y_pred,batch_out.to(device))
            # params update
            optm.zero_grad()
            loss_out.backward()
            optm.step()

            loss_val_sum+=loss_out
        loss_val_avg=loss_val_sum/len(train_iter)
        if (epoch%print_every==0) or (epoch==EPOCHS-1) :
            train_accr=func_eval(Model,train_iter)
            vali_accr=func_eval(Model,vali_iter)
            print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] vali_accr:[%.3f]"%(epoch,loss_val_avg,train_accr,vali_accr))
        else :
            vali_accr=func_eval(Model,vali_iter)
            print("epoch:[%d] loss:[%.3f] train_accr:[skip] vali_accr:[%.3f]"%(epoch,loss_val_avg,vali_accr))

        if vali_accr>best_accr:
            best_accr=vali_accr
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            print('Model saved...\nBest_acc : [%.3f]\n'%(best_accr))
            torch.save(Model,f'{checkpoint_dir}/model.pt')
            torch.save(Model.state_dict(),f'{checkpoint_dir}/model_state_dict.pt')
            torch.save({'model':Model.state_dict(),
                        'optm':optm.state_dict(),
                        'best_accuracy':best_accr},f'{checkpoint_dir}/all.tar')
    print("Training Done.")
    return

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_data():
    rgb_mean = (0.5,)
    rgb_std = (0.5,)
    train_transforms=transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize(rgb_mean,rgb_std),
                                        transforms.RandomApply([AddGaussianNoise(0.,1.)],p=0.2)])
    vali_transforms=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(rgb_mean,rgb_std)])
    #train_data=DatasetFashion('./datasets/train.csv',test_mode=False,train=True,transform=train_transforms)
    vali_data=DatasetFashion('./datasets/train.csv',test_mode=False,train=False,transform=vali_transforms)
    #TEST MODE ---------
    train_data=DatasetFashion('./datasets/train.csv',test_mode=True,train=True,transform=train_transforms)
    test_data=DatasetFashion('./datasets/test.csv',test_mode=True,train=False,transform=vali_transforms)
    #-------------------
    train_iter=torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
    vali_iter=torch.utils.data.DataLoader(vali_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
    test_iter=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=1)
    return train_iter,vali_iter,test_iter

def image_show(image, ax=None,title=None,normalize=True) :
    if ax is None:
        flag, ax = plt.subplots()
    image=image.numpy().transpose((1,2,0))

    if normalize:
        # Normalize Image -> Origin Image
        mean=0.5
        std=0.5
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

def test_show(Model,vali_iter) :
    images, labels = next(iter(vali_iter))
    # I want to show image (sample=25)
    n_sample=25
    test_x=images[:n_sample]
    test_y=labels[:n_sample]

    # get Model pred by Sample image
    with torch.no_grad():
        Model.eval()
        y_pred = Model.forward(test_x.view(-1,1,28,28).type(torch.float).to(device))
        y_pred = y_pred.reshape(-1,10)
        Model.train()
    y_pred=y_pred.argmax(axis=1)

    # Image show
    ordered=["T-Shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
    plt.figure(figsize=(20,20))
    for idx in range(n_sample):
        ax=plt.subplot(5,5,idx+1)
        title=f"Predict:{ordered[y_pred[idx]]}, Label:{ordered[test_y[idx]]}"
        image_show(test_x[idx],ax,title,normalize=True)
    plt.show()

def main() :
    best_accr=0
    train_iter, vali_iter, test_iter=get_data()
    if os.path.exists(checkpoint_dir):
        print("Loading Prev Model Setting.....")
        Model=torch.load(f'{checkpoint_dir}/model.pt')
        Model.load_state_dict(torch.load(f'{checkpoint_dir}/model_state_dict.pt'))
        checkpoint=torch.load(f'{checkpoint_dir}/all.tar')
        best_accr=checkpoint['best_accuracy']
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        optm.load_state_dict(checkpoint['optm'])
        Model.to(device)
    else :
        print("New Model Setting.....")
        Model=CNN(name='cnn',imageSize=[28,28],input_channel=1,start_channels=[32,64],block_channels=[128,256],block_loof=2,out_channel=10).to(device)
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
    loss=nn.CrossEntropyLoss()

    print("Model Setting Done.")
    count_parameters(Model)
    #model_train(Model,train_iter,vali_iter,optm,loss,best_accr)
    #test_show(Model,vali_iter)
    last_test(Model,test_iter)
    test_show(Model,test_iter)

if __name__ == "__main__":
    main()
    