from data_processing import *
import torch
import torch.utils.data as data
import torch.optim as optim
from collections import defaultdict
from model import *
import gensim
import os
# CONFIG
torch.manual_seed(777)
BATCH_SIZE=100
EPOCHS=10
LEARNING_RATE=0.001
MAX_PAD_LEN=10
vocab=TrainWord("./datasets/train.csv",1)
vocab.build_vocab()
vocab_size=vocab.vocab_size
#DEVICE
print(f'PyTorch Version : [{torch.__version__}]')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device : [{device}]')

def get_model() :
    #Model=NormalLinearModel(device,vocab_size,[512,256,128],5)
    Model=GruModel(device,vocab_size,5)
    return Model

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

def get_data() :
    train_data=DatasetNovel("./datasets/train.csv",vocab,test=False,train=True)
    vali_data=DatasetNovel("./datasets/train.csv",vocab,test=False,train=False)
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

def get_embedding():
    print("\nConvert to Embedding layer....\n")
    embedding_layer=[]
    zero_layer=[0.]*300
    word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin',binary=True)
    for word,index in tqdm(vocab.vocab_to_idx.items()):
        try :
            embedding_layer.append(word2vec_model.wv[word])
        except KeyError:
            embedding_layer.append(zero_layer)
    embedding_layer=torch.FloatTensor(embedding_layer)
    print("\nDone...\n")
    print(f'Embedding length : {embedding_layer.size()}')
    print(f'Vocab size : {vocab.vocab_size}')
    print("\nSave embedding layer to tensor...\n")
    torch.save(embedding_layer,'embedding.pt')
    print("\nDone...\n")

def main():
    if os.path.exists('./embedding.pt')==False:
        get_embedding()
    train_iter, vali_iter=get_data()
    Model=get_model()
    Model.to(device)
    Model.init_params()
    loss=nn.CrossEntropyLoss()
    optm=optim.Adam(Model.parameters(),lr=LEARNING_RATE)
    print(Model)
    train(Model,train_iter,vali_iter,optm,loss)

if __name__ == "__main__":
    main()
