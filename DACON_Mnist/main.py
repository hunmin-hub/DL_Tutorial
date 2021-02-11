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
import os
from models import EnsembleModel, LinearModel, CnnModel, CustomModel
# CONFIG
torch.manual_seed(1204)
BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=0.001
checkpoint_dir="./weights"
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def get_Model():
    Model_A = LinearModel(in_channel=28*28,linear_channel=[256,64,32,10])
    Model_B = CnnModel(image_size=[28,28],in_channel=1,hidden_channel=[32,64,32,10])
    Model_C = CustomModel(image_size=[28,28],in_channel=1,block1_channel=[64,32,64],block2_channel=[64,32,64],out_channel=10)
    Model=EnsembleModel(Model_A,Model_B,Model_C)
    return Model

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
        print("(Train or Validation) Data Testing....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            y_target=batch_out.to(device)
            model_pred=Model(batch_in.view(-1,28*28).to(device),batch_in.view(-1,1,28,28).to(device),batch_in.view(-1,1,28,28).to(device))
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
            y_pred=Model(batch_in.view(-1,28*28).to(device),batch_in.view(-1,1,28,28).to(device),batch_in.view(-1,1,28,28).to(device))
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
                        'optimizer':optm.state_dict(),
                        'best_accuracy':best_accr},f'{checkpoint_dir}/all.tar')

    print("Training Done.")
    return

class DatasetMnist(data.Dataset):
    def __init__(self,file_path,test_mode=False,train=True,transform=None):
        self.file_path=file_path
        self.test_mode=test_mode
        self.train=train
        self.transform=transform
        data=pd.read_csv(self.file_path,index_col='index')
        self.dataframe=list(data.values)

        image=[]
        label=[]
        if test_mode==False :
            #Train -> Get train.csv -> train,vali data
            train_len=int(len(self.dataframe)*0.8)
            if self.train :
                #Train
                for idx in range(0,train_len):
                    label.append(self.dataframe[idx][0])
                    image.append(self.dataframe[idx][1:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
            else :
                #Vali
                for idx in range(train_len,len(self.dataframe)):
                    label.append(self.dataframe[idx][0])
                    image.append(self.dataframe[idx][1:])
                self.labels=np.asarray(label)
                self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
        else :
            # Final Test Mode = (Train + Vali)-> ALL Train / Test data(No label) -> Test data
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
        if self.transform != None:
            image=self.transform(image)
        return image,label

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
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize(rgb_mean,rgb_std),
                                        transforms.RandomApply([AddGaussianNoise(0.,1.)],p=0.5)])
    vali_transforms=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(rgb_mean,rgb_std)])
    #train_data=DatasetMnist('./datasets/train.csv',test_mode=False,train=True,transform=train_transforms)
    train_data=DatasetMnist('./datasets/train.csv',test_mode=True,train=True,transform=train_transforms)
    vali_data=DatasetMnist('./datasets/train.csv',test_mode=False,train=False,transform=vali_transforms)
    test_data=DatasetMnist('./datasets/test.csv',test_mode=True,train=False,transform=vali_transforms)
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
        y_pred = Model(test_x.view(-1,28*28).to(device),test_x.view(-1,1,28,28).to(device),test_x.view(-1,1,28,28).to(device))
        y_pred = y_pred.reshape(-1,10)
        Model.train()
    y_pred=y_pred.argmax(axis=1)

    # Image show
    ordered=["0","1","2","3","4","5","6","7","8","9"]
    plt.figure(figsize=(20,20))
    for idx in range(n_sample):
        ax=plt.subplot(5,5,idx+1)
        title=f"Predict:{ordered[y_pred[idx]]}, Label:{ordered[test_y[idx]]}"
        image_show(test_x[idx],ax,title,normalize=True)
    plt.show()

## Last Test to csv
def last_test(Model,test_iter):
    pred_label=[]
    with torch.no_grad():
        Model.eval()
        print("Test Data Testing.....\n")
        for batch_in, batch_out in tqdm(iter(test_iter)):
            model_pred=Model(batch_in.view(-1,28*28).to(device),batch_in.view(-1,1,28,28).to(device),batch_in.view(-1,1,28,28).to(device))
            model_pred=model_pred.reshape(-1,10)
            _, y_pred=torch.max(model_pred.data,1) # My Model pred
            pred_label.extend(y_pred.tolist())
    print("Testing done.\n")
    print("Result to submission.csv")
    submission = pd.read_csv('./datasets/sample_submission.csv', encoding = 'utf-8')
    submission['label'] = pred_label
    submission.to_csv('./datasets/mnist_submission.csv', index = False)
    return

def main() :
    best_accr=0
    # Get data
    train_iter, vali_iter, test_iter=get_data()
    # Model Setting
    if os.path.exists(checkpoint_dir):
        print("Loading Prev Model Setting.....\n")
        Model=torch.load(f'{checkpoint_dir}/model.pt')
        Model.load_state_dict(torch.load(f'{checkpoint_dir}/model_state_dict.pt'))
        checkpoint=torch.load(f'{checkpoint_dir}/all.tar')
        best_accr=checkpoint['best_accuracy']
        optimizer=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer'])
        Model.to(device)
    else :
        print("New Model Setting.....\n")
        Model=get_Model()
        optimizer=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        Model.to(device)
    criterion=nn.CrossEntropyLoss()
    print(Model)
    count_parameters(Model)
    # Train
    #model_train(Model,train_iter,vali_iter,optimizer,criterion,best_accr)
    last_test(Model,test_iter)
    test_show(Model,test_iter)

if __name__ == "__main__" :
    main()