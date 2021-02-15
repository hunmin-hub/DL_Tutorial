import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from generator import GeneratorModel
from discriminator import DiscriminatorModel
from tqdm import tqdm
#CONFIG
BATCH_SIZE=200
LEARNING_RATE=0.001
EPOCHS=100
z_size=72
checkpoint_dir='./weights'
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def get_model():
    Model_G=GeneratorModel(z_size,in_channel=1)
    Model_D=DiscriminatorModel(in_channel=1,hidden_channels=[8,16,32])
    return Model_G, Model_D

def count_parameters(Model) :
    total_params = 0
    for param_name, param in Model.named_parameters():
        if param.requires_grad:
            print(f'[{param_name}] Layers Parameters : [{len(param.reshape(-1))}] shape : [{param.size()}]')
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")

def get_data():
    face_transforms=transforms.Compose([transforms.Resize((28,28)),
                                        transforms.ToTensor(),
                                        transforms.Grayscale(),
                                        transforms.Normalize((0.5,),(0.5,))])
    face_data=datasets.ImageFolder('./datasets',transform=face_transforms)
    face_iter=torch.utils.data.DataLoader(face_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
    return face_iter

def image_show(images,epoch):
    N_SAMPLE=16
    plt.figure(figsize=(20,20))
    sample_image=images[:N_SAMPLE]
    for idx in range(N_SAMPLE):
        ax=plt.subplot(4,4,idx+1)
        current_image=sample_image[idx].cpu().numpy().transpose((1,2,0))
        #mean=0.5
        #std=0.5
        #current_image=std*current_image+mean
        current_image=np.clip(current_image,0,1)
        ax.imshow(current_image)
    plt.savefig('./{:03d}.jpg'.format(14+epoch))

def model_train(Model_G,Model_D,face_iter,G_optimizer,D_optimizer):
    criterion = nn.BCEWithLogitsLoss()    
    Model_G.train()
    Model_D.train()
    for epoch in range(EPOCHS):
        for real_face, _ in tqdm(iter(face_iter)):
            real_face=real_face.view(-1,1,28,28).to(device)
            C_BATCH_SIZE=len(real_face)
            real_labels = torch.ones(C_BATCH_SIZE).to(device) # Real : 1
            fake_labels = torch.zeros(C_BATCH_SIZE).to(device) # Fake : 0
            model_pred=Model_D(real_face)
            D_loss_real=criterion(model_pred,real_labels)
            real_score=torch.sigmoid(model_pred)

            z=torch.randn(C_BATCH_SIZE,z_size,1,1).to(device) #잠재벡터 Z
            fake_face=Model_G(z)

            model_pred=Model_D(fake_face.detach())
            D_loss_fake=criterion(model_pred,fake_labels)

            fake_score=torch.sigmoid(model_pred)

            D_loss=D_loss_real+D_loss_fake

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            model_pred=Model_D(fake_face)
            G_loss=criterion(model_pred,real_labels)

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

        print('\nEpoch[{:3d}/{:3d}] D_loss: {:.4f}, G_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}\n'.format(
        epoch, EPOCHS, D_loss.item(), G_loss.item(), real_score.mean().item(), fake_score.mean().item()))
        
        with torch.no_grad():
            Model_G.eval()
            z=torch.randn(BATCH_SIZE,z_size,1,1).to(device)
            fake_face=Model_G(z)
            image_show(fake_face,epoch)
            Model_G.train()
        
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        print('Model Saved...')
        torch.save(Model_G,f'{checkpoint_dir}/model_G.pt')
        torch.save(Model_G.state_dict(),f'{checkpoint_dir}/model_G_state_dict.pt')
        torch.save({'model':Model_G.state_dict(),
                    'optimizer':G_optimizer.state_dict()},f'{checkpoint_dir}/G_all.tar')
        torch.save(Model_D,f'{checkpoint_dir}/model_D.pt')
        torch.save(Model_D.state_dict(),f'{checkpoint_dir}/model_D_state_dict.pt')
        torch.save({'model':Model_D.state_dict(),
                    'optimizer':D_optimizer.state_dict()},f'{checkpoint_dir}/D_all.tar')

def main() :
    
    if os.path.exists(checkpoint_dir):
        print("Model Loading...\n")
        Model_G=torch.load(f'{checkpoint_dir}/model_G.pt')
        Model_G.load_state_dict(torch.load(f'{checkpoint_dir}/model_G_state_dict.pt'))
        G_checkpoint=torch.load(f'{checkpoint_dir}/G_all.tar')

        Model_D=torch.load(f'{checkpoint_dir}/model_D.pt')
        Model_D.load_state_dict(torch.load(f'{checkpoint_dir}/model_D_state_dict.pt'))
        D_checkpoint=torch.load(f'{checkpoint_dir}/D_all.tar')

        D_optimizer = optim.Adam(Model_D.parameters(), lr=LEARNING_RATE)
        G_optimizer = optim.Adam(Model_G.parameters(), lr=LEARNING_RATE)
        D_optimizer.load_state_dict(D_checkpoint['optimizer'])
        G_optimizer.load_state_dict(G_checkpoint['optimizer'])

    else :
        Model_G, Model_D=get_model()
        Model_G.init_params()
        Model_D.init_params()
        D_optimizer = optim.Adam(Model_D.parameters(), lr=LEARNING_RATE)
        G_optimizer = optim.Adam(Model_G.parameters(), lr=LEARNING_RATE)
    
    Model_G.to(device)
    Model_D.to(device)
    count_parameters(Model_G)
    count_parameters(Model_D)
    face_iter=get_data()
    model_train(Model_G,Model_D,face_iter,G_optimizer,D_optimizer)

    with torch.no_grad():
        Model_G.eval()
        z=torch.randn(BATCH_SIZE,z_size,1,1).to(device)
        fake_face=Model_G(z)
        image_show(fake_face)

if __name__ == "__main__":
    main()
