import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from models import EnsembleModel,ModelA,ModelB,ModelC,ModelD,ModelE
from tqdm import tqdm
import os
#CONFIG
torch.manual_seed(1201)
BATCH_SIZE=10
EPOCHS=50
LEARNING_RATE=1e-3
checkpoint_dir="./weights"
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

# ----------------------- Model Setting ------------------------------------ #
def get_model() :
    # EnsembleModel INFO
    # 1. Model_A : Normal CNN Model (Filter Size=3)
    # 2. Model_B : Normal CNN Model (Filter Size=5)
    # 3. Model_C : Custom ResNet Model (Filter Size=3)
    # 4. Model_D : Custom ResNet Model (Filter Size=5)
    # 5. Model_E : Custom DenseNet Model (Filter Size=3)
    Model_A = ModelA(image_size=128,in_channel=3,hidden_channels=[32,64,128],output_channel=6)
    Model_B = ModelB(image_size=128,in_channel=3,hidden_channels=[32,64,128],output_channel=6)
    Model_C = ModelC(image_size=128,in_channel=3,block_channels=[64,128,256],output_channel=6)
    Model_D = ModelD(image_size=128,in_channel=3,block_channels=[64,128,256],output_channel=6)
    Model_E = ModelE(image_size=128,in_channel=3,hidden_channels=[32,64,128],out_channel=6)
    Model=EnsembleModel(Model_A,Model_B,Model_C,Model_D,Model_E)
    return Model

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
            model_pred=Model(batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device))
            model_pred=model_pred.reshape(-1,6)
            loss_out=loss(model_pred,batch_out.to(device))
            loss_val_sum+=loss_out
            _, y_pred=torch.max(model_pred.data,1)
            n_correct+=(y_pred==y_target).sum().item()
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        Model.train()
        loss_val_avg=loss_val_sum/len(data_iter)
    print("Testing Done.\n")
    return val_accr,loss_val_avg

# ----------------------- Model Train ------------------------------------ #
def model_train(Model,optm,loss,best_accr):
    data_iters = get_data() # New K-FOLD
    if best_accr==0 :
        Model.init_params()
    else :
        Model.eval()
    Model.train()
    print("Start Training....\n")
    for idx,(train_iter, vali_iter) in enumerate(data_iters):
        print(f'\n K Fold Step : [{idx+1}/4] Training.....\n')
        for epoch in range(EPOCHS):
            loss_val_sum=0
            for batch_in, batch_out in tqdm(iter(train_iter)):
                y_pred=Model(batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device))
                y_pred=y_pred.reshape(-1,6)
                loss_out=loss(y_pred,batch_out.to(device))
                # params update
                optm.zero_grad()
                loss_out.backward()
                optm.step()
                loss_val_sum+=loss_out

            loss_val_avg=loss_val_sum/len(train_iter)
            vali_accr,vali_loss=func_eval(Model,vali_iter,loss)
            print("epoch:[%d] train loss:[%.3f] vali loss:[%.3f] vali_accr:[%.3f]\n"%(epoch,loss_val_avg,vali_loss,vali_accr))

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

def get_data() :
    train_transforms=transforms.Compose([transforms.Resize(150),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(128),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    dataset=datasets.ImageFolder('./datasets/train',transform=train_transforms)
    folds=KFold(n_splits=4,shuffle=True)
    data_iters=[]
    for i_fold,(train_idx,vali_idx) in enumerate(folds.split(dataset)):
        train_data=torch.utils.data.Subset(dataset,train_idx)
        vali_data=torch.utils.data.Subset(dataset,vali_idx)

        train_iter=torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
        vali_iter=torch.utils.data.DataLoader(vali_data,batch_size=BATCH_SIZE,shuffle=True)

        data_iters.append((train_iter,vali_iter))
    return data_iters # K-FOLD (Train data + Vali data)

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
    ordered=['Apple','Banana','Durian','KoreanMelon','Mandarin','Strawberry']
    plt.figure(figsize=(20,20))
    count=0
    print("Processing to Test Image..")
    for images, labels in tqdm(iter(data_iter)):
        test_x=images[:BATCH_SIZE]
        test_y=labels[:BATCH_SIZE]
        for idx in range(BATCH_SIZE):
            if count==96 : break
            ax=plt.subplot(10,10,count+1)
            if pred_label[count]!=test_y[idx]:
                title=f'X'
            else :
                title=f'O'
            image_show(test_x[idx],ax,title,normalize=True)
            count+=1
    plt.show()

def last_test(Model):
    # Get Test Set
    test_transforms=transforms.Compose([transforms.Resize(150),
                                        transforms.CenterCrop(128),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_data=datasets.ImageFolder('./datasets/test',transform=test_transforms)
    test_iter=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

    pred_label=[]
    test_correct=0
    total=0
    with torch.no_grad():
        Model.eval()
        print("Test Data Testing.....\n")
        for batch_in, batch_out in tqdm(iter(test_iter)):
            y_target=batch_out.to(device) # True Target
            model_pred=Model(batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device),
                            batch_in.view(-1,3,128,128).to(device))
            model_pred=model_pred.reshape(-1,6)
            _, y_pred=torch.max(model_pred.data,1) # My Model pred
            pred_label.extend(y_pred.tolist())
            test_correct+=(y_pred==y_target).sum().item()
            total+=batch_in.size(0)
        test_accr=(test_correct/total)
    print("Testing done.\n")
    print("Final Test accuracy is [%.3f]\n"%(test_accr))
    sample_show(test_iter,pred_label)
    return

def main() :
    best_accr=0
    if os.path.exists(checkpoint_dir):
        print("Loading Prev Model Setting.....\n")
        Model=torch.load(f'{checkpoint_dir}/model.pt')
        Model.load_state_dict(torch.load(f'{checkpoint_dir}/model_state_dict.pt'))
        checkpoint=torch.load(f'{checkpoint_dir}/all.tar')
        best_accr=checkpoint['best_accuracy']
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
        optm.load_state_dict(checkpoint['optimizer'])
    else :
        print("New Model Setting.....\n")
        Model=get_model()
        optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
    loss=nn.CrossEntropyLoss()
    Model.to(device)

    count_parameters(Model)

    #model_train(Model,optm,loss,best_accr)
    last_test(Model)

if __name__ == "__main__" :
    main()
