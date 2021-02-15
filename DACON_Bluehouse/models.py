from collections import defaultdict
from tqdm import tqdm
import math

class NaiveBayesClassifier():
    def __init__(self,w2i,k=0.1):
        self.k=k
        self.w2i=w2i # Train data -> tokens -> word to index (token to index)
        self.priors={}
        self.likelihoods={}

    def train(self,train_tokenized,train_labels):
        self.set_priors(train_labels)
        self.set_likelihoods(train_tokenized, train_labels)
    
    def inference(self,tokens):
        log_prob=[0.0]*3

        for token in tokens :
            if token in self.likelihoods:
                for idx in range(3):
                    log_prob[idx]+=math.log(self.likelihoods[token][idx])
        
        for idx in range(3):
            log_prob[idx]+=math.log(self.priors[idx])
        
        return log_prob.index(max(log_prob))

    def set_priors(self,train_labels):
        class_counts=defaultdict(int)
        for label in tqdm(train_labels):
            class_counts[label]+=1
        
        for label, count in class_counts.items():
            self.priors[label]=class_counts[label]/len(train_labels)
    
    def set_likelihoods(self,train_tokenized,train_labels):
        token_dists=defaultdict(lambda:defaultdict(int))
        class_counts=defaultdict(int)
        print("\nCalculate likelihoods......\n")
        for idx,label in tqdm(enumerate(train_labels)):
            count=0
            for token in train_tokenized[idx]:
                if token in self.w2i :
                    token_dists[token][label]+=1
                    count+=1
            class_counts[label]+=count
        
        for token, dist in tqdm(token_dists.items()):
            if token not in self.likelihoods:
                self.likelihoods[token]={
                    0:(token_dists[token][0]+self.k)/(class_counts[0]+len(self.w2i)*self.k),
                    1:(token_dists[token][1]+self.k)/(class_counts[1]+len(self.w2i)*self.k),
                    2:(token_dists[token][2]+self.k)/(class_counts[2]+len(self.w2i)*self.k)
                }
        print("\nDone...\n")
                

            
            
