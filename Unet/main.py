import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from model import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# CONFIG
torch.manual_seed(1204)
BATCH_SIZE=4
EPOCHS=10
LEARNING_RATE=0.001
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def image_show(input,label,output,epoch):
    result_dir='./result'
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    label = fn_tonumpy(label)
    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
    output = fn_tonumpy(fn_class(output))

    for j in range(label.shape[0]):
        id = 10*epoch+j+1

        plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
        plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
        plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

        np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
        np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
        np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

def func_eval(Model,data_iter,loss,epoch):
    with torch.no_grad():
        Model.eval()
        loss_val_sum=0
        print("(Train or Validation) Data Testing....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            batch_in=batch_in.float()
            batch_out=batch_out.float()
            y_target=batch_out.to(device)
            y_target=y_target.float()
            model_pred=Model(batch_in.view(-1,1,512,512).to(device))
            loss_out=loss(model_pred,y_target)
            loss_val_sum+=loss_out
            image_show(batch_in,batch_out,model_pred,epoch)
        Model.train()
        loss_val_avg=loss_val_sum/len(data_iter)
    print("Testing Done.\n")
    return loss_val_avg

def train(Model,train_iter,vali_iter,optm,loss):
    Model.train()
    print("Start Training...\n")
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)):
            batch_in=batch_in.float()
            batch_out=batch_out.float()
            y_target=batch_out.to(device)
            y_target=y_target.float()
            y_pred=Model(batch_in.view(-1,1,512,512).to(device))
            loss_out=loss(y_pred,y_target)
            # params update
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum+=loss_out

        loss_val_avg=loss_val_sum/len(train_iter)
        vali_loss=func_eval(Model,vali_iter,loss,epoch)
        print("epoch:[%d] train loss:[%.3f] vali loss:[%.3f]\n"%(epoch,loss_val_avg,vali_loss))

def get_model():
    Model=Unet()
    return Model

def get_data():
    rgb_mean = (0.5,)
    rgb_std = (0.5,)
    custom_transforms=transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(rgb_mean,rgb_std)])

    train_data=CellDataset('./datasets/train',custom_transforms)
    vali_data=CellDataset('./datasets/vali',custom_transforms)

    train_iter=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    vali_iter=DataLoader(vali_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    return train_iter,vali_iter

def main():
    train_iter, vali_iter = get_data()
    Model=get_model()
    Model.to(device)
    print(Model)
    Model.init_params()

    loss=nn.BCEWithLogitsLoss().to(device)
    optm=torch.optim.Adam(Model.parameters(),lr=LEARNING_RATE)

    train(Model,train_iter,vali_iter,optm,loss)

if __name__ == "__main__":
    main()

