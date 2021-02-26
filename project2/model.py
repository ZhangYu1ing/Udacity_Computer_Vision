import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         pass
        super().__init__()
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        input_size = embed_size
        bias = True
        batch_first = True # the input and output tensors are provided as (batch, seq, feature)
        dropout = 0
        bidirectional = False
#         proj_size = 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

   
        
        
    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device = device), torch.zeros((1, batch_size, self.hidden_size), device = device))
          

    
    def forward(self, features, captions):
#         pass
        
        # initialize 
        
        self.hidden = self.init_hidden(features.shape[0])
        
        # remove <end> from captions
        word_embed = self.word_embeddings(captions[:, :-1])
        
        word_embed = torch.cat((features.unsqueeze(1), word_embed), dim= 1)
        
        out_lstm, self.hidden = self.lstm(word_embed, self.hidden)
        
        return self.linear(out_lstm)
    



    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         pass
        output = []
        hidden = self.init_hidden(inputs.shape[0])
        pos = 0
        
        while pos < max_len:
            out_lstm, hidden = self.lstm(inputs, hidden)
            out = self.linear(out_lstm)
            out = out.squeeze(1)
            
            a, max_index = torch.max(out, dim = 1)
            
            output.append(max_index.item())
            
            inputs = self.word_embeddings(max_index)
            inputs = inputs.unsqueeze(1)
            
            pos += 1
            if max_index == 1:
                break;
        
        return output
            
            
        
        
        
        