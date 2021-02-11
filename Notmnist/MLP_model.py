from torchvision import datasets, transforms, models, utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
EPOCHS=4
BATCH_SIZE=256

print(f'PyTorch version:[{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device:[{device}]')

class MultiLayerPerceptronClass(nn.Module): # MLP
    def __init__(self,name='MLP',xdim=784,hdim=256,ydim=10):
        super(MultiLayerPerceptronClass,self).__init__()
        self.name=name
        self.xdim=xdim
        self.hdim=hdim
        self.ydim=ydim
        self.lin_1 = nn.Linear(self.xdim,self.hdim)
        self.lin_2 = nn.Linear(self.hdim,self.ydim)
        self.init_param()
    
    def init_param(self):
        nn.init.kaiming_normal_(self.lin_1.weight)
        nn.init.zeros_(self.lin_1.bias)
        nn.init.kaiming_normal_(self.lin_2.weight)
        nn.init.zeros_(self.lin_2.bias)
    
    def forward(self,x):
        net=x
        net=self.lin_1(net)
        net=F.relu(net)
        net=self.lin_2(net)
        return net

def get_data():
    data_transforms=transforms.Compose([transforms.ToTensor(),
                                        transforms.Grayscale(num_output_channels=1)])
    train_data=datasets.ImageFolder('./dataset/train',transform=data_transforms)
    test_data=datasets.ImageFolder('./dataset/test',transform=data_transforms)
    # print(train_data.classes) LABEL 목록 -> list
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    print("Data setting done....\n")
    return train_loader,test_loader,train_data.classes

def func_eval(model,data_iter,device):
    with torch.no_grad():
        model.eval()
        n_total, n_correct=0,0
        print("Testing.....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)) :
            y_target=batch_out.to(device)
            model_pred=model(batch_in.view(-1,28*28).to(device))
            _, y_pred=torch.max(model_pred.data,1)
            n_correct+=(y_pred==y_target).sum().item()
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        model.train()
    return val_accr

def model_train(model,train_iter,test_iter,optm,loss) :
    model.init_param()
    model.train()
    print("Start Training...\n")
    print_every=1
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)) :
            y_pred=model.forward(batch_in.view(-1,28*28).to(device))
            loss_out=loss(y_pred,batch_out.to(device))
            # parameters update
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum+=loss_out
        loss_val_avg=loss_val_sum/len(train_iter)
        if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
            train_accr=func_eval(model,train_iter,device)
            test_accr=func_eval(model,test_iter,device)
            print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]"%(epoch,loss_val_avg,train_accr,test_accr))
    print("Training Done.")
    return

def get_param(model) :
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer Parameters : {len(param.reshape(-1))}")
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")

def main() :
    train_loader,test_loader,label_list=get_data()
    M=MultiLayerPerceptronClass(name='MLP',xdim=784,hdim=256,ydim=10).to(device)
    loss=nn.CrossEntropyLoss()
    optm=optim.Adam(M.parameters(),lr=1e-3)
    model_train(M,train_loader,test_loader,optm,loss)
    get_param(M)

if __name__ == "__main__" :
    main()

