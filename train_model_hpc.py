import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from transformers import BertModel, BertConfig

SEED = 42

import warnings
warnings.filterwarnings('ignore')

training_data = pd.read_csv('data/X_train.csv')
labels = pd.read_csv('data/y_train.csv')

# drop rows with missing text data
training_data = training_data.dropna(subset=['text'])
labels = labels.loc[training_data.index]

# removing punctation
def remove_puncation(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return text

training_data['text'] = training_data['text'].apply(remove_puncation)

# removing html tags if they exist
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

training_data["text"] = training_data["text"].apply(remove_tags)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True).to_device("cuda:0")

def encode_data(data):

    input_ids = []
    attention_masks = []
    
    for text in data["text"]:
        encoded_dict = tokenizer.encode_plus(
                            text,                     
                            add_special_tokens = True,
                            truncation = "longest_first", # only sequences that are truncated are the ones that really need it due to their excessive length
                            max_length = 74,         
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            return_tensors = 'pt',
                    )
        
        # add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])
        
        # add the attention mask
        attention_masks.append(encoded_dict['attention_mask'])

    # convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

train_input_ids, train_attention_masks = encode_data(training_data)

# reseting the indices of the DataFrame
training_data = training_data.reset_index(drop=True)

# converting labels into tensors
labels = torch.tensor(labels.label)

#train and validation sets split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_input_ids, labels, random_state=SEED, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(train_attention_masks, labels, random_state=SEED, test_size=0.2)

def build_model():
    # configuring the BERT model for binary classification.
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=1,  # indicates that we are doing binary classification
        output_attentions=False,  # specifies if the model returns attentions weights.
        output_hidden_states=False,  # specifies if the model returns all hidden-states.
    )
    
    # loading the pre-trained BERT model 
    bert_layer = BertModel.from_pretrained("bert-base-uncased", config=config)

    class BertForBinaryClassification(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = bert_layer  # the BERT model layer
            self.dropout = nn.Dropout(0.1)  # dropout layer to prevent overfitting
            self.classifier = nn.Linear(config.hidden_size, 1)  # a linear layer for classification

        def forward(self, input_ids, attention_mask=None):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]  # the pooled_output is the representation of the [CLS] token
            drop_output = self.dropout(pooled_output)  # applying dropout to the [CLS] token representation
            logits = self.classifier(drop_output)  # pass through the classifier to get the prediction logits
            return logits

    return BertForBinaryClassification()

model = build_model()

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)

batch_size = 32

# dataLoader for training and validation sets
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=batch_size
)

# initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# since this is a binary classification problem, we use BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()

# number of epochs to train
num_epochs = 5  # we can always change this and play around
best_val_loss = float('inf')


raise Exception("ALL GOOD")
# training loop

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = batch

        # clear previous gradients
        model.zero_grad()

        # perform a forward pass, the forward method will return the logits
        logits = model(b_input_ids, attention_mask=b_input_mask)

        # compute loss and perform a backward pass
        loss = loss_fn(logits.view(-1), b_labels.float())  # we can adjust the shape of logits and labels if necessary
        loss.backward()
        total_train_loss += loss.item()

        # update parameters
        optimizer.step()

    # print training loss after each epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.3f}')

    # validation phase
    
    model.eval() # evaluation mode
    total_val_loss = 0
    
    with torch.no_grad():  # deactivating autograd for evaluation
        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(logits.view(-1), b_labels.float())  # computing validation loss
            total_val_loss += loss.item()

    # calculating average validation loss
    avg_val_loss = total_val_loss / len(validation_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_val_loss:.3f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.bin')
        print(f"Saved model with validation loss {avg_val_loss:.3f}")