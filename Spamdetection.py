import time

import pandas
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch import nn, optim, tensor
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_labels, dropout):
        super(Classifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.dropout = dropout

        self.dp = nn.Dropout(self.dropout)
        self.ff = nn.Linear(self.embedding_dim, self.num_labels)

    def forward(self, input_embeddings):
        tensor = self.dp(input_embeddings)
        tensor = self.ff(tensor)
        return tensor, F.softmax(tensor, dim=-1)


class Batcher(object):
    def __init__(self, data_x, data_y, batch_size):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.n_samples = data_x.shape[0]
        self.indices = torch.randperm(self.n_samples)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr > self.n_samples:
            self.ptr = 0
            self.indices = torch.randperm(self.n_samples)
            raise StopIteration
        else:
            batch_indices = self.indices[self.ptr:self.ptr+self.batch_size]
            self.ptr += self.batch_size
            return self.data_x[batch_indices], self.data_y[batch_indices]


class Spam():
    def __init__(self, filename):
        self.path = filename
    
    def read_and_filtercsv(self):
        input_df = pandas.read_csv(self.path)
             
        input_df.dropna(inplace=True)
        input_df.drop_duplicates(inplace=True)
        input_df.rename(columns={
            "target": "labels",
            "Message": "text"
        }, inplace=True)
        return input_df
    
    def encode_and_transform(self,input_df):
        le = LabelEncoder()
        le.fit(input_df.labels)
        input_df["labels"] = le.transform(input_df.labels)
        
        train_x, test_x, train_y, test_y = \
            train_test_split(input_df.text, input_df.labels, stratify=input_df.labels, test_size=0.15,
                             random_state=123)
        
        train_x_de, test_x_de, train_y_de, test_y_de = \
            train_test_split(input_df.text_fr, input_df.labels, stratify=input_df.labels, test_size=0.15,
                             random_state=123)
    
        sentences = train_x.tolist()
        test_sentences = test_x_de.tolist()
        
        self.labels = torch.tensor(train_y.tolist())
        self.test_labels = torch.tensor(test_y_de.tolist())
        
        # encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
        encoder = SentenceTransformer('quora-distilbert-multilingual')
        print('Encoding segments...')
        start = time.time()
        self.embedding = encoder.encode(sentences, convert_to_tensor=True)
        self.test_sentences_embedding = encoder.encode(test_sentences, convert_to_tensor=True)
        print(f"Encoding completed in {time.time() - start} seconds.")
    
    
    def train_model(self):
        train_batcher = Batcher(self.embedding, self.labels, batch_size=16)
        
        num_samples, embeddings_dim = self.embedding.size()
        n_labels = self.labels.unique().shape[0]
        
        self.classifier = Classifier(embeddings_dim, n_labels, dropout=0.01)
        
        optimizer = optim.Adam(self.classifier.parameters())
        loss_fn = nn.CrossEntropyLoss()
        
        for e in range(10):
            total_loss = 0
            for batch in train_batcher:
                x, y = batch
                optimizer.zero_grad()
                self.classifier = self.classifier.to(device)
                model_output, prob = self.classifier(x)
                model_output = model_output.to('cpu')
                loss = loss_fn(model_output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #print(f'epoch:{e}, total_loss:{total_loss}')
    
    def test_model(self):
        with torch.no_grad():
            model_output, prob = self.classifier(self.test_sentences_embedding)
            model_output = model_output.to('cpu')
            prob = prob.to('cpu')
            predictions = torch.argmax(prob, dim=-1)
            results = classification_report(predictions, self.test_labels)
            print(results)
            

if __name__ == "__main__":
    obj = Spam("SPAM text message 20170820-fr.csv")
    input_df = obj.read_and_filtercsv()
    obj.encode_and_transform(input_df)
    obj.train_model()
    obj.test_model()