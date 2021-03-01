import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalLinearModel(nn.Module):
    def __init__(self,device,vocab_size,hidden_dim=[512,256,128],output_dim=5):
        super(NormalLinearModel,self).__init__()
        self.device=device
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=2
        self.num_dirs=2
        #self.embedding_layer=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=256,padding_idx=0)
        embedding_weight=torch.load('./embedding.pt')
        self.embedding_layer=nn.Embedding.from_pretrained(embedding_weight,padding_idx=0)
        prev_dim=300
        hidden_layer=[]
        for current_dim in self.hidden_dim:
            hidden_layer.append(nn.Linear(prev_dim,current_dim))
            hidden_layer.append(nn.ReLU(inplace=True))
            hidden_layer.append(nn.Dropout(p=0.2))
            prev_dim=current_dim
        self.hidden_net=nn.Sequential(*hidden_layer)
        self.output_layer=nn.Linear(prev_dim,output_dim)
    
    def forward(self,x):
        x=self.embedding_layer(x)
        x=self.hidden_net(torch.sum(x,1))
        x=self.output_layer(x)
        return x
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

class GruModel(nn.Module):
    def __init__(self,device,vocab_size,output_dim=5):
        super(GruModel,self).__init__()
        self.device=device
        self.vocab_size=vocab_size
        self.output_dim=output_dim
        self.num_layers=1
        self.num_dirs=1
        embedding_weight=torch.load('./embedding.pt')
        self.embedding_layer=nn.Embedding.from_pretrained(embedding_weight,padding_idx=0)
        #self.embedding_layer=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=300,padding_idx=0)
        self.GRU=nn.GRU(input_size=300,hidden_size=300,num_layers=self.num_layers,bidirectional=False)
        prev_dim=300*self.num_dirs
        self.output_layer=nn.Linear(prev_dim,output_dim)
    
    def forward(self,x):
        x=self.embedding_layer(x)
        h_0=torch.zeros((self.num_layers*self.num_dirs,x.size(0),x.size(-1))).to(self.device)
        x,_ = self.GRU(x.transpose(0,1),h_0)
        x=x[-1]
        x=self.output_layer(x)
        return x
    
    def init_params(self) :
        for m in self.modules() :
            if isinstance(m,nn.Linear) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)