import pandas as pd
import numpy as np
import spacy
import warnings
import re
import random
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")
warnings.filterwarnings('ignore')

columns = ['Index','Country','Sentiment','Text']
training_data = pd.read_csv('twitter_training.csv',names = columns)
validation_data = pd.read_csv('twitter_validation.csv', names=columns)


training_data.drop_duplicates(inplace=True)
training_data.dropna(inplace=True)


def clean_emoji(tx):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols 
                               u"\U0001F680-\U0001F6FF"  # transport 
                               u"\U0001F1E0-\U0001F1FF"  # flags 
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', tx)


def text_cleaner(tx):
    text = re.sub(r"won\'t", "would not", tx)
    text = re.sub(r"im", "i am", tx)
    text = re.sub(r"Im", "I am", tx)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"needn\'t", "need not", text)
    text = re.sub(r"hasn\'t", "has not", text)
    text = re.sub(r"haven\'t", "have not", text)
    text = re.sub(r"weren\'t", "were not", text)
    text = re.sub(r"mightn\'t", "might not", text)
    text = re.sub(r"didn\'t", "did not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\!\?\.\@]', ' ', text)
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[@]+', '@', text)
    text = re.sub(r'unk', ' ', text)
    text = re.sub('\n', '', text)
    text = text.lower()
    text = re.sub(r'[ ]+', ' ', text)

    return text

random.seed(99)
test_text =text_cleaner( random.choice(training_data['Text']))
test_text = clean_emoji(test_text)
test_text

doc = nlp(test_text)
for chunk in doc.ents:
    print(f'{chunk} => {chunk.label_}')

doc = nlp(test_text)
for chunk in doc.noun_chunks:
    print(f'{chunk} => {chunk.label_}')

Tokenizer=RegexpTokenizer(r'\w+')
test_text_tokenized = Tokenizer.tokenize(test_text)
test_text_tokenized



stopwords_list = stopwords.words('english')
print(f'There are {len(stopwords_list) } stop words')
print('**' * 20 , '\n20 of them are as follows:\n')
for inx , value in enumerate(stopwords_list[:20]):
    print(f'{inx+1}:{value}')

def make_corpus(kind):
    corpus = []
    for text in training_data.loc[training_data['Sentiment']==kind]['Text'].str.split():
        for word in text:
            corpus.append(word)
    return corpus


stop = stopwords.words('english')
sentiments = list(training_data.Sentiment.unique())

for inx, value in enumerate(sentiments):

    corpus = make_corpus(value)

    dic = defaultdict(int)

    for word in corpus:
        if word in stop:
            dic[word] += 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]

    x, y = zip(*top)
    plt.title(f'{value} ')
    plt.bar(x, y)
    plt.show()

training_data['Text'] = training_data['Text'].apply(lambda x : text_cleaner(x))
training_data['Text']= training_data['Text'].apply(lambda x : Tokenizer.tokenize(x))
training_data['Text'].to_frame()

nlp = spacy.load("en_core_web_sm")
doc = nlp(test_text)
for token in doc :
    print(f'{token} => {token.lemma_}')

Stemmer = PorterStemmer()
def stopwords_cleaner(text):
#     word = [lemmatizer.lemmatize(letter) for letter in text if letter not in stopwords_list]
    word = [Stemmer.stem(letter) for letter in text if letter not in stopwords_list]
    peasting = ' '.join(word)
    return peasting
training_data['Text'] = training_data['Text'].apply(lambda x : stopwords_cleaner(x))
# stopwords_cleaner(Tokenizer.tokenize(df.Text[100]))

training_data['sentiments'] = training_data['Sentiment'].replace({'Positive' : 1 ,  'Negative' : 0 ,'Neutral':2 , 'Irrelevant' : 2 })

positive_reviews = training_data[training_data['Sentiment'] == 'Positive']['Text']
pos = ' '.join(map(str, positive_reviews))
pos_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                         stopwords=stopwords_list,
                          min_font_size=15).generate(pos)
plt.figure(figsize=(10, 10))
plt.imshow(pos_wordcloud)
plt.title('Word Cloud for Positive Reviews')
plt.axis('off')
plt.show()

negative_reviews = training_data[training_data['Sentiment'] == 'Negative']['Text']
neg = ' '.join(map(str, negative_reviews))
pos_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                         stopwords=stopwords_list,
                          min_font_size=15).generate(neg)
plt.figure(figsize=(10, 10))
plt.imshow(pos_wordcloud)
plt.title('Word Cloud for Negative Reviews')
plt.axis('off')
plt.show()

positive_reviews = training_data[training_data['Sentiment'] == 'Neutral']['Text']
Neutral = ' '.join(map(str, positive_reviews))
pos_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                         stopwords=stopwords_list,
                          min_font_size=15).generate(Neutral)
plt.figure(figsize=(10, 10))
plt.imshow(pos_wordcloud)
plt.title('Word Cloud for Neutral Reviews')
plt.axis('off')
plt.show()

positive_reviews = training_data[training_data['Sentiment'] == 'Irrelevant']['Text']
Irrelevant  = ' '.join(map(str, positive_reviews))
pos_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                         stopwords=stopwords_list,
                          min_font_size=15).generate(Irrelevant )
plt.figure(figsize=(10, 10))
plt.imshow(pos_wordcloud)
plt.title('Word Cloud for Irrelevant Reviews')
plt.axis('off')
plt.show()

class Dataset:
    def __init__(self, text, sentiment):
        self.text = text
        self.sentiment = sentiment

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item, :]
        target = self.sentiment[item]
        return {
            "text": torch.tensor(text, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long)
        }


def load_vectors(fname):
    fin = open(fname, encoding="utf8")
    data = {}
    for line in fin:
        tokens = line.split()
        data[tokens[0]] = np.array([float(value) for value in tokens[1:]])

    return data


def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]

    return embedding_matrix


class sentimentBiLSTM(nn.Module):
    # inherited from nn.Module

    def __init__(self, embedding_matrix, hidden_dim, output_size):
        # initializing the params by initialization method
        super(sentimentBiLSTM, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        num_words = self.embedding_matrix.shape[0]
        embed_dim = self.embedding_matrix.shape[1]
        # craetinh embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)

        ## initializes the weights of the embedding layer to the pretrained embeddings in
        ## embedding_matrix. It first converts embedding_matrix to a PyTorch tensor and
        ## wraps it in an nn.Parameter object, which makes it a learnable parameter of the model.
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # it is multuplied by 2 becuase it is bi_directional if one-sided it didnt need.
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    # we need a forward function to model calculate the cost and know how bad the params is .
    # However , it can be written in a line of code but if we want to track it it is easier way.
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1]
        out = self.fc(lstm_out)

        return out

y = training_data.sentiments.values
train_df,test_df = train_test_split(training_data,test_size = 0.2, stratify = y)

MAX_LEN = 167
BATCH_SIZE = 32
hidden_dim = 64
output_size = 3

if torch.cuda.is_available():

    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print(f'Current device is {device}')

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_data.Text.values.tolist())

xtrain = tokenizer.texts_to_sequences(train_df.Text.values)
xtest = tokenizer.texts_to_sequences(test_df.Text.values)
xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain,maxlen = MAX_LEN)
xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest,maxlen = MAX_LEN)
train_dataset = Dataset(text=xtrain,sentiment=train_df.sentiments.values)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,drop_last=True)
valid_dataset = Dataset(text=xtest,sentiment=test_df.sentiments.values)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,drop_last=True)


embedding_dict = load_vectors('/Users/sachinlk/Desktop/Python basics/Python Project/glove.6B.300d.txt')
embedding_matrix = create_embedding_matrix(tokenizer.word_index,embedding_dict)

model = sentimentBiLSTM(embedding_matrix ,  hidden_dim, output_size)
model = model.to(device)
print(model)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
# schedul_learning = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer , milestones=[6] ,
#                                                         gamma=0.055)

def acc(pred,label):
    pred = pred.argmax(1)
    return torch.sum(pred == label.squeeze()).item()


clip = 5
epochs = 9
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(epochs):
    # for getting loss and accuracy for train
    train_losses = []
    train_acc = 0.0

    # put model on train mode
    model.train()
    correct = 0

    # initialize hidden state
    for data in train_loader:
        # get text and target
        inputs = data['text']
        labels = data['target']

        # put them on GPU and right dtypes
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        # gradient becomes zero=> avoid accumulating
        model.zero_grad()
        output = model(inputs)
        # calculate the loss and perform backprop
        loss = criterion(output, labels.long())
        loss.backward()
        train_losses.append(loss.item())
        # accuracy
        accuracy = acc(output, labels)
        train_acc += accuracy
        # `clip_grad_norm` helps prevent the exploding gradient problem in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    # for getting loss and accuracy for valiadtion
    val_losses = []
    val_acc = 0.0

    # put model on evaluation mode
    model.eval()
    for data in valid_loader:
        # get text and target
        inputs = data['text']
        labels = data['target']

        # put them on GPU and right dtypes
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        # gradient becomes zero=> avoid accumulating
        model.zero_grad()
        output = model(inputs)

        output = model(inputs)
        # Loss calculating
        val_loss = criterion(output, labels.long())
        # append Loss to the above list
        val_losses.append(val_loss.item())

        # calculating accuracy
        accuracy = acc(output, labels)
        val_acc += accuracy
        epoch_train_loss = np.mean(train_losses)

        # using schedule lr if you need
    #         schedul_learning.step()
    #         schedul_learning

    # appending all accuracy and loss to the above lists and variables
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
    if epoch_val_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min} --> {epoch_val_loss})  Saving model ...')
        # save model if better result happends
        valid_loss_min = epoch_val_loss
    print(30 * '==', '>')

plt.figure(figsize=(7,5))
plt.plot(range(1,10),epoch_tr_acc , label='train accuracy')
plt.scatter(range(1,10),epoch_tr_acc)
plt.plot(range(1,10),epoch_vl_acc , label='val accuracy')
plt.scatter(range(1,10),epoch_vl_acc)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(range(1,10),epoch_tr_loss , label='train loss')
plt.scatter(range(1,10),epoch_tr_loss )
plt.plot(range(1,10),epoch_vl_loss , label='val loss')
plt.scatter(range(1,10),epoch_vl_loss)
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

new_sequences = tokenizer.texts_to_sequences(validation_data['Text'])
new_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=MAX_LEN)

# Convert to PyTorch tensor
new_data_tensor = torch.tensor(new_padded_sequences, dtype=torch.long).to(device)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert predictions to labels (assuming the model outputs numerical predictions)
predicted_labels = predictions.argmax(dim=1).cpu().numpy()


# Add predicted labels to the validation_data DataFrame
validation_data['Predicted_Sentiment'] = predicted_labels

# Display or save the DataFrame with predicted sentiments
print(validation_data[['Text', 'Predicted_Sentiment']])