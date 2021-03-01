from transformers import *
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import random
random.seed(777)
#CONFIG
torch.manual_seed(777)
BATCH_SIZE=50
EPOCHS=10
MAX_PAD_LEN=10
LEARNING_RATE=0.00001
#BERT
bert_name='bert-base-uncased'
config = BertConfig.from_pretrained(bert_name)
tokenizer = BertTokenizer.from_pretrained(bert_name,do_lower_case=True)
Bert_Model = BertModel.from_pretrained(bert_name)
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

class BertClassificationModel(nn.Module):
    def __init__(self,Model,output_dim=5):
        super(BertClassificationModel,self).__init__()
        self.output_dim=output_dim
        self.Model=Model
        self.output_layer=nn.Linear(768,self.output_dim)
    
    def forward(self,x):
        last_hidden, CLS_x=self.Model(x) # 마지막 hidden state 값, Classification을 위한 CLS 토큰
        x=self.output_layer(CLS_x)
        return x

class DatasetNovel(Dataset):
    def __init__(self,file_path,test,train):
        self.file_path=file_path
        self.test=test
        self.train=train

        self.df=pd.read_csv(self.file_path,index_col='index')
        self.df=list(self.df.values)
        self.sentence=[]
        if self.test==False:
            self.train_len=int(len(self.df)*0.8)
            if self.train :
                for idx in range(0,self.train_len):
                    self.sentence.append([self.df[idx][0],self.df[idx][1]])
            else :
                for idx in range(self.train_len,len(self.df)):
                    self.sentence.append([self.df[idx][0],self.df[idx][1]])
        else :
            for idx in range(len(self.df)):
                self.sentence.append([self.df[idx][0],0])

    def __getitem__(self,index):
        return tokenizer.encode(self.sentence[index][0],max_length=100,truncation=True),self.sentence[index][1]
    
    def __len__(self):
        return len(self.sentence)

def bucketed_batch_indices(sentence_length,batch_size,max_pad_len):
    batch_map=defaultdict(list)
    batch_indices_list=[]
    len_min=min(sentence_length,key=lambda x:x[0])[0]
    for idx, (length,label) in enumerate(sentence_length):
        src=(length-len_min+1)//(max_pad_len)
        batch_map[src].append(idx)
    
    for key,value in batch_map.items():
        batch_indices_list+=[value[i:i+batch_size] for i in range(0,len(value),batch_size)]
    
    random.shuffle(batch_indices_list)
    return batch_indices_list

def collate_fn(batched_samples):
    PAD=0
    batch_size=len(batched_samples)
    
    batched_samples=sorted(batched_samples,key=lambda x:len(x[0]),reverse=True)
    src_sentences=[]
    labels=[]
    for sentence,label in batched_samples:
        src_sentences.append(torch.tensor(sentence))
        labels.append(label)
    
    src_sentences=torch.nn.utils.rnn.pad_sequence(src_sentences,batch_first=True)
    return src_sentences,torch.tensor(labels)

def get_data():
    train_data=DatasetNovel("./datasets/train.csv",test=False,train=True)
    vali_data=DatasetNovel("./datasets/train.csv",test=False,train=False)
    # Train
    print("\nTrain data setting....\n")
    sentence_length=list(map(lambda pair:(len(pair[0]),pair[1]),train_data))
    bucketed_batch_indices(sentence_length,batch_size=BATCH_SIZE,max_pad_len=MAX_PAD_LEN)
    train_iter=data.dataloader.DataLoader(train_data,collate_fn=collate_fn,num_workers=2,
                                        batch_sampler=bucketed_batch_indices(sentence_length,batch_size=BATCH_SIZE,max_pad_len=MAX_PAD_LEN))
    print("\nDone..\n")
    # Vali
    print("\nVali data setting....\n")
    sentence_length=list(map(lambda pair:(len(pair[0]),pair[1]),vali_data))
    bucketed_batch_indices(sentence_length,batch_size=BATCH_SIZE,max_pad_len=MAX_PAD_LEN)
    vali_iter=data.dataloader.DataLoader(vali_data,collate_fn=collate_fn,num_workers=2,
                                        batch_sampler=bucketed_batch_indices(sentence_length,batch_size=BATCH_SIZE,max_pad_len=MAX_PAD_LEN))
    print("\nDone..\n")
    return train_iter, vali_iter

def func_eval(Model,data_iter):
    with torch.no_grad():
        Model.eval()
        n_total,n_correct=0,0
        print("\nTesting....\n")
        for batch_in, batch_out in tqdm(iter(data_iter)):
            y_target=batch_out.to(device)
            model_pred=Model(batch_in.to(device))
            _, y_pred=torch.max(model_pred.data,1)
            n_correct+=(y_pred==y_target).sum().item()
            n_total+=batch_in.size(0)
        val_accr=(n_correct/n_total)
        Model.train()
    print("Testing Done.\n")
    return val_accr

def train(Model,train_iter,vali_iter,optm,loss):
    Model.train()
    print("\nStart Model Training...\n")
    for epoch in range(EPOCHS):
        loss_val_sum=0
        for batch_in, batch_out in tqdm(iter(train_iter)):
            model_pred=Model(batch_in.to(device))
            loss_out=loss(model_pred,batch_out.to(device))
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum+=loss_out
        loss_val_avg=loss_val_sum/len(train_iter)
        train_accr=func_eval(Model,train_iter)
        vali_accr=func_eval(Model,vali_iter)
        print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] vali_accr:[%.3f]"%(epoch,loss_val_avg,train_accr,vali_accr))
    print("\nTraining Done..\n")
    return

def main():
    train_iter, vali_iter=get_data()

    Model=BertClassificationModel(Bert_Model,5)
    Model.to(device)
    loss=nn.CrossEntropyLoss()
    optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
    train(Model,train_iter,vali_iter,optm,loss)


if __name__ == "__main__":
    main()