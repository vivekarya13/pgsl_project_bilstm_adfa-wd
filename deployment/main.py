import os
import gensim
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from math import ceil
from data_loader import load_data
from data_preprocessing import data_processor
from model import lstm
from train import trainer
from evaluate import tester
from predictor import predictor


# train and test path
train_path = '/data/ADFA-WD/Full_Trace_Training_Data'
attack_path = '/data/ADFA-WD/Full_Trace_Attack_Data'
val_path = '/data/ADFA-WD/Full_Trace_Validation_Data'


N_GRAM=3

# loading the data
train_sequences,train_labels   = load_data(train_path  , n=N_GRAM)
val_sequences,val_labels     = load_data(val_path    , n=N_GRAM)
attack_sequences,attack_labels  = load_data(attack_path , n=N_GRAM)

# checking for a sequence
print('Sequence length : ', len(train_sequences))
print('Train labels: ',train_labels)

# prepare the dataset
x_train,y_train,x_test,y_test,max_seq,num_token, embed_size,embed_matrix = data_processor(train_sequences,train_labels,val_sequences,val_labels,attack_sequences,attack_labels)

# model
model = lstm(max_seq,num_token,embed_size,embed_matrix)
model.summary()

# training
trainer(model,x_train,y_train)

# testing
tester(model,x_test,y_test)
