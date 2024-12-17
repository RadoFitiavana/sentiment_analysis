import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def getLSTMClassifier(
        X,
        vocab_size = 10000,
        embedding_dim = 50,
        hidden_size = 128, 
        dropout_rate = 0.5,
        learning_rate = 0.001,
        epochs = 10,
        batch_size = 64
):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X.shape[1]),
        LSTM(hidden_size, return_sequences=False), 
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model