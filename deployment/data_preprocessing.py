# preprocessing of the data
from gensim.models import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from math import ceil
import copy


def word_vec(train):
    model=Word2Vec(train, vector_size=100, window=5, min_count=1)
    return model

def ngram_to_indices(ngram, word2vec_dict,model):
    # Compute the average vector of all available vectors in the dictionary
    all_vectors = [vec for vec in word2vec_dict.values()]
    avg_vector = np.mean(all_vectors, axis=0) if all_vectors else np.zeros(word2vec_dict.vector_size)

    # Use the average vector for unknown tokens
    return [word2vec_dict.get(token, avg_vector) for token in ngram]




def embed_n_pad(corpus_train_indices,corpus_test_indices,word2vec_dict,model):
    embedding_size = 100 # As per your Word2Vec model
    num_tokens = len(word2vec_dict) # Number of unique tokens in the dictionary
    print(num_tokens)
    embedding_matrix = np.zeros((num_tokens, embedding_size))

    # Populate the embedding matrix with Word2Vec embeddings
    for word, idx in word2vec_dict.items():
        embedding_matrix[idx] = model.wv[word]

    # Prepare the data: pad sequences to ensure uniform input size
    max_sequence_length = max(len(sentence) for sentence in corpus_train_indices)
    X_train = pad_sequences(corpus_train_indices, maxlen=max_sequence_length, value=-1, padding="post")
    X_test = pad_sequences(corpus_test_indices, maxlen=max_sequence_length, value=-1, padding="post")
    return X_train,X_test,max_sequence_length,embedding_size,embedding_matrix


def data_processor(train_sequences,train_labels,val_sequences,val_labels,attack_sequences,attack_labels):
    sattack_lists = {i: [] for i in range(13)}
    # Populate the dictionary based on integer values
    for attack_num, sublist in zip(attack_labels, attack_sequences):
        if attack_num in sattack_lists:  # Check if the value is in the desired range
            sattack_lists[attack_num].append(sublist)

    from math import ceil

    # Dictionaries to hold the split data for each key
    attack_train_data = {}
    attack_val_data = {}
    attack_test_data = {}

    # Split each list of lists in data_dict
    # we can update this logic
    for key, data in sattack_lists.items():
        total_length = len(data)
        train_size = ceil(total_length * 0.7) #Breaking test data in 70 and 30
        test_size = total_length - train_size
        '''
        train_size = ceil(total_length * 0.6)
        val_size = ceil(total_length * 0.2)
        test_size = total_length - train_size - val_size
        '''
        # Split the data and store in the corresponding dictionaries
        attack_train_data[key] = data[:train_size]
        attack_test_data[key] = data[train_size:]

        '''
        attack_train_data[key] = data[:train_size]
        attack_val_data[key] = data[train_size:train_size + val_size]
        attack_test_data[key] = data[train_size + val_size:]
        '''

    T_DATLIST = []
    T_LABLIST = []
    ftrain_sequences = []
    ftrain_labels = []
    ftrain_sequences = copy.deepcopy(train_sequences)
    ftrain_labels = train_labels[:]

        # Iterate over each key and its list of lists
    for key, lists in attack_train_data.items():
        print(key)
        # Extend merged_data with each sublist in the current key's list of lists
        T_DATLIST.extend(lists)
        # Extend key_labels with the key repeated for each sublist
        T_LABLIST.extend([key] * len(lists))

    ftrain_sequences.extend(T_DATLIST)
    ftrain_labels.extend(T_LABLIST)

    T_DATLIST = []
    T_LABLIST = []
    ftest_sequences = []
    ftest_labels = []

    # Final test data created from validation data + 30% of attack data
    lv_VAL_LEN = len(val_sequences)
    lv_VAL_SIZ = ceil(lv_VAL_LEN * 0.5) #Breaking val data in 50 %
    ftest_sequences = copy.deepcopy(val_sequences[:lv_VAL_SIZ])
    ftest_labels    = val_labels[:lv_VAL_SIZ]

    # Iterate over each key and its list of lists
    for key, lists in attack_test_data.items():
        print(key)
        # Extend merged_data with each sublist in the current key's list of lists
        T_DATLIST.extend(lists)
        # Extend key_labels with the key repeated for each sublist
        T_LABLIST.extend([key] * len(lists))

    ftest_sequences.extend(T_DATLIST)
    ftest_labels.extend(T_LABLIST)

    m=word_vec(ftrain_sequences)
    word2vec_dict = m.wv.key_to_index

    corpus_train_indices = [ngram_to_indices(sentence, word2vec_dict,m) for sentence in ftrain_sequences]
    corpus_test_indices = [ngram_to_indices(sentence, word2vec_dict,m) for sentence in ftest_sequences]
    num_tokens = len(word2vec_dict)
    ftrain,ftest,max,embed_size,embed_mat = embed_n_pad(corpus_train_indices,corpus_test_indices,word2vec_dict,m)

    X_train = np.array(ftrain)  # Convert list to NumPy array
    X_test = np.array(ftest)  # Convert list to NumPy array
    y_train = to_categorical(ftrain_labels) # Convert to one-hot encoding
    y_test  = to_categorical(ftest_labels)
    return X_train,y_train,X_test,y_test,max,num_tokens,embed_size,embed_mat
