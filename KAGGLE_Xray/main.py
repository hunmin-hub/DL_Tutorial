import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import DenseModel
# CONFIG
BATCH_SIZE=5
LEARNING_RATE=0.001
EPOCHS=50
checkpoint_dir='./weights'
# Device
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def count_parameters(Model) :
    total_params = 0
    for param_name, param in Model.named_parameters():
        if param.requires_grad:
            print(f'[{param_name}] Layers Parameters : [{len(param.reshape(-1))}] shape : [{param.size()}]')
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")

def func_eval(Model,data_iter,loss):
    with torch.no_grad():
        Model.eval()
        n_total, n_correct = 0,0
        loss_val_sum=0
        print("(Train or Validation) Data Testing....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            y_target=batch_out.to(device)
            y_target=y_target.float()
            model_pred=Model(batch_in.view(-1,1,224,224).to(device))
            n_correct+=((torch.sigmoid(model_pred)>=0.5)==y_target).sum().item()
            loss_out=loss(model_pred,y_target)
            loss_val_sum+=loss_out
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        Model.train()
        loss_val_avg=loss_val_sum/len(data_iter)
    print("Testing Done.\n")
    return val_accr,loss_val_avg

# ----------------------- Model Train ------------------------------------ #
def model_train(Model,train_iter,vali_iter,optm,loss,best_loss):
    Model.train()
    print("Start Training....\n")
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)):
            y_target=batch_out.to(device)
            y_target=y_target.float()
            y_pred=Model(batch_in.view(-1,1,224,224).to(device))
            loss_out=loss(y_pred,y_target)
            # params update
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum+=loss_out

        loss_val_avg=loss_val_sum/len(train_iter)
        vali_accr,vali_loss=func_eval(Model,vali_iter,loss)
        print("epoch:[%d] train loss:[%.3f] vali loss:[%.3f] vali_accr:[%.3f]\n"%(epoch,loss_val_avg,vali_loss,vali_accr))
        if vali_loss<best_loss:
            best_loss=vali_loss
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            print('Model saved...\nBest_loss : [%.3f]\n'%(best_loss))
            torch.save(Model,f'{checkpoint_dir}/model.pt')
            torch.save(Model.state_dict(),f'{checkpoint_dir}/model_state_dict.pt')
            torch.save({'model':Model.state_dict(),
                        'optimizer':optm.state_dict(),
                        'best_loss':best_loss},f'{checkpoint_dir}/all.tar')

    print("Training Done.")
    return

def get_data() :
    data_transforms=transforms.Compose([transforms.Resize(250),
                                        transforms.CenterCrop(224),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
    dataset=datasets.ImageFolder('./datasets/train',transform=data_transforms)
    train_data_len=int(len(dataset)*0.8)
    test_data=datasets.ImageFolder('./datasets/test',transform=data_transforms)
    # Split (train, vali)
    train_data, vali_data=torch.utils.data.random_split(dataset,[train_data_len,len(dataset)-train_data_len])
    # Get train,vali,test iter
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
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
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

def sample_show(data_iter,pred_label):
    plt.figure(figsize=(20,20))
    count=0
    current_label=0
    ordered=["X","O"]
    for images, labels in tqdm(iter(data_iter)):
        test_x=images[:BATCH_SIZE]
        test_y=labels[:BATCH_SIZE]
        for idx in range(BATCH_SIZE):
            if count==40 : break
            ax=plt.subplot(7,7,count+1)
            if pred_label[count] :
                current_label=1
            else :
                current_label=0
            title=f'Model : {ordered[current_label]} True : {ordered[test_y[idx]]}'
            image_show(test_x[idx],ax,title,normalize=False)
            count+=1
    plt.show()

def last_test(Model,test_iter):
    with torch.no_grad():
        Model.eval()
        print("Last Test........\n")
        pred_label=[]
        test_correct=0
        total=0
        for batch_in, batch_out in tqdm(iter(test_iter)):
            y_target=batch_out.to(device)
            model_pred=Model(batch_in.to(device))
            pred_list=(torch.sigmoid(model_pred)>=0.5)
            pred_label.extend(pred_list)
            test_correct+=(pred_list==y_target).sum().item()
            total+=batch_in.size(0)
        test_accr=(test_correct/total)
        print("Testing done.\n")
        print("Final Test accuracy is [%.3f]\n"%(test_accr))
    sample_show(test_iter,pred_label)

def main() :
    best_loss=99999
    train_iter,vali_iter,test_iter=get_data()
    if os.path.exists(checkpoint_dir):
        print("Loading Prev Model Setting.....\n")
        Model=torch.load(f'{checkpoint_dir}/model.pt')
        Model.load_state_dict(torch.load(f'{checkpoint_dir}/model_state_dict.pt'))
        checkpoint=torch.load(f'{checkpoint_dir}/all.tar')
        best_loss=checkpoint['best_loss']
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        optm.load_state_dict(checkpoint['optimizer'])
        Model.to(device)
    else :
        print("New Model Setting.....\n")
        Model=DenseModel(image_size=224,in_channel=1,block_count=7,out_channel=1,k=12)
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        Model.to(device)
        Model.init_params()
    loss=nn.BCEWithLogitsLoss()
    count_parameters(Model)
    
    #-model_train(Model,train_iter,vali_iter,optm,loss,best_loss)
    last_test(Model,test_iter)

if __name__ == "__main__":
    main()