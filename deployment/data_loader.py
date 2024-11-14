# to load the dataset
import os
import gensim
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# label maps
label_map = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4,
             'P1': 5, 'P2': 6, 'P3': 7, 'P4': 8,
             'S1': 9, 'S2': 10, 'S3': 11, 'S4': 12,
             'Train': 0, 'Valid': 0
             }

def generate_ngrams(sequence, n):
    #generate n-grams from a sequence as strings of n words.
    return [' '.join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def load_data(dataset_path, n):
    sequences = []
    labels = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == '.DS_Store':
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                #print(file)
                #print(root)
                #print(file_path)
                sequence = f.read().strip().split()
                '''all.extend(sequence)'''
                n_grams = generate_ngrams(sequence, n)
                sequences.append(n_grams)

                for Lab_substring, label in label_map.items():
                    if Lab_substring in root:
                      labels.append(label)
                      break

    return sequences, labels
