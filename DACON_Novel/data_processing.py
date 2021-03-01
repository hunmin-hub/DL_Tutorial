import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter,defaultdict
import random
import re
random.seed(777)

def processing(text):
    text=re.sub('[^a-zA-Z]',' ',text)
    # word to Tokenize
    tokens=[word for sentence in nltk.sent_tokenize(text)
                for word in nltk.word_tokenize(sentence)]
    
    # remove stop words
    stop=stopwords.words('english')
    tokens=[token for token in tokens if token not in stop]
    
    # word to lower
    tokens=[word.lower() for word in tokens]

    # Lenma 표제어 추출
    lmtzr = WordNetLemmatizer()
    tokens=[lmtzr.lemmatize(word,'v') for word in tokens]

    # stemming
    stemmer=PorterStemmer()
    tokens=[stemmer.stem(word) for word in tokens]
    
    return tokens

class TrainWord(Dataset):
    # Build Vocab 구축
    PAD_TOKEN='<PAD>'
    PAD_TOKEN_IDX=0
    UNK_TOKEN='<UNK>'
    UNK_TOKEN_IDX=1
    EOS_TOKEN='<EOS>'
    EOS_TOKEN_IDX=2
    SEP_TOKEN='<SEP>'
    SEP_TOKEN_IDX=3
    CLS_TOKEN='<CLS>'
    CLS_TOKEN_IDX=4
    def __init__(self,file_path,min_freq=1):
        '''
        1. Sentence to Token
        2. Build Vocab
        '''
        self.file_path=file_path
        self.min_freq=min_freq
        self.train_word=[]
        # Load Train, Vali, Test
        self.train_data=pd.read_csv(self.file_path,index_col='index')
        self.train_data=list(self.train_data.values)
        self.train_len=int(len(self.train_data)*0.8)
    
    def build_vocab(self):
        SPECIAL_TOKENS=[TrainWord.PAD_TOKEN,TrainWord.UNK_TOKEN,TrainWord.EOS_TOKEN,TrainWord.SEP_TOKEN,TrainWord.CLS_TOKEN]
        print("\nTrain word to build vocab....\n")
        print("\nTrain word to processing....\n")
        for idx in tqdm(range(0,self.train_len)):
            self.train_word.extend(processing(self.train_data[idx][0]))
        print("\nProcessing Train word to idx_to_vocab....\n")
        self.idx_to_vocab=SPECIAL_TOKENS+[word for word,count in tqdm(Counter(self.train_word).items()) if count>=self.min_freq]
        print("\nidx_to_vocab to vocab_to_idx.........\n")
        self.vocab_to_idx={word : idx for idx, word in tqdm(enumerate(self.idx_to_vocab))}
        print("\nDone.\n")
        self.vocab_size=len(self.idx_to_vocab)

class DatasetNovel(Dataset):
    def __init__(self,file_path,vocab,test,train):
        self.file_path=file_path
        self.vocab=vocab
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
        return preprocess(self.vocab,self.sentence[index][0]),self.sentence[index][1]
    
    def __len__(self):
        return len(self.sentence)
        

def preprocess(vocab,text):
    text=' '.join(processing(text))
    sentence=[TrainWord.CLS_TOKEN_IDX]
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if word in vocab.vocab_to_idx :
                sentence.append(vocab.vocab_to_idx[word])
            else :
                sentence.append(TrainWord.UNK_TOKEN_IDX)
        sentence.append(TrainWord.SEP_TOKEN_IDX)
    sentence.append(TrainWord.EOS_TOKEN_IDX)
    return sentence

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
    PAD=TrainWord.PAD_TOKEN_IDX
    batch_size=len(batched_samples)
    
    batched_samples=sorted(batched_samples,key=lambda x:len(x[0]),reverse=True)
    src_sentences=[]
    labels=[]
    for sentence,label in batched_samples:
        src_sentences.append(torch.tensor(sentence))
        labels.append(label)
    
    src_sentences=torch.nn.utils.rnn.pad_sequence(src_sentences,batch_first=True)
    return src_sentences,torch.tensor(labels)



