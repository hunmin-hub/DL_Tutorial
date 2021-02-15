import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from models import NaiveBayesClassifier

class DatasetBluehouse(Dataset):
    def __init__(self,file_path,test_mode=False,train=True):
        self.file_path=file_path
        self.test_mode=test_mode
        self.train=train

        data=pd.read_csv(self.file_path,index_col='index',encoding='utf-8')
        data["data"]=data["data"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","") # 한글과 공백 제외 모두 제거
        self.dataframe=list(data.values)
        self.sentence=[]
        self.labels=[]
        if self.test_mode==False :
            self.train_len=int(len(self.dataframe)*0.8)
            if self.train :
                for idx in range(self.train_len):
                    self.labels.append(self.dataframe[idx][0])
                    self.sentence.append(self.dataframe[idx][1])
            else :
                for idx in range(self.train_len,len(self.dataframe)):
                    self.labels.append(self.dataframe[idx][0])
                    self.sentence.append(self.dataframe[idx][1])
        else :
            if self.train :
                for idx in range(len(self.dataframe)):
                    self.labels.append(self.dataframe[idx][0])
                    self.sentence.append(self.dataframe[idx][1])
            else :
                for idx in range(len(self.dataframe)):
                    self.labels.append(0)
                    self.sentence.append(self.dataframe[idx][0])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        label=self.labels[index]
        sent=self.sentence[index]
        return sent,label

def get_data():
    train_data=DatasetBluehouse('./datasets/train.csv',test_mode=True,train=True)
    test_data=DatasetBluehouse('./datasets/test.csv',test_mode=True,train=False)

    return train_data, test_data

def data_processing(train_data,vali_data):
    stopwords=['의','가','이','은','다','들','을','는','인','위해','과','던','도','를','로','게','으로','까지','자','에','을까','는데','치','와','한','하다']
    tokenizer=Mecab()
    train_tokens=[]
    vali_tokens=[]
    # 형태소 분석
    print("Train data : Word to token....\n")
    for sentence in tqdm(train_data.sentence):
        token_text=[word for word in tokenizer.morphs(str(sentence)) if not word in stopwords]
        train_tokens.append(token_text)
    print("Done.\n")
    print("Vali(Test) data : Word to token....\n")
    for idx,sentence in tqdm(enumerate(vali_data.sentence)):
        token_text=[word for word in tokenizer.morphs(str(sentence)) if not word in stopwords]
        vali_tokens.append(token_text)
    print("Done.\n")
    # Make Vocab
    print("Make Vocab....\n")
    word_count=defaultdict(int)
    for tokens in tqdm(train_tokens):
        for token in tokens:
            word_count[token]+=1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    print(f"word_count len : {len(word_count)}\nDone.\n")
    # Word to index
    print("Make Word to index...\n")
    w2i={}
    for pair in tqdm(word_count):
        if pair[0] not in w2i:
            w2i[pair[0]]=len(w2i)
    print("Done.\n")

    return train_tokens, vali_tokens, w2i
    
def main():
    train_data, test_data = get_data()
    train_tokens, test_tokens, w2i = data_processing(train_data,test_data)
    #Train
    classifier = NaiveBayesClassifier(w2i)
    classifier.train(train_tokens,train_data.labels)
    preds=[]
    print("Testing..........\n")
    for tokens in tqdm(test_tokens):
        preds.append(classifier.inference(tokens))
    submission = pd.read_csv('./datasets/sample_submission.csv', encoding = 'utf-8')
    submission['category'] = preds
    submission.to_csv('./datasets/Bluehouse_submission.csv', index = False)

if __name__ == "__main__":
    main()



        
