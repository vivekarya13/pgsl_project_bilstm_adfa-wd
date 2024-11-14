from keras.layers import Embedding, Masking, Bidirectional, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam

def lstm(max_sequence_length,num_tokens,embedding_size,embedding_matrix):
    model=Sequential()
    model.add(Masking(mask_value=-1, input_shape=(max_sequence_length,)))
    model.add(Embedding(input_dim=num_tokens,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],  # Load pre-trained embedding matrix
                            input_length=max_sequence_length,   # Specify the sequence length
                            trainable=False,             # Set to True if you want to fine-tune
    ))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Add Dense layer for multiclass classification
    model.add(Dense(13, activation='softmax'))
    return model
