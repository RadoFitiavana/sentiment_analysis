import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils import *

print("Preprocessing data ...\n")

df = pd.read_csv('BodoPoopy.csv')
df['cleaned_text'] = df['text'].apply(clean_text)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['Label'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

X = padded_sequences
y = encoded_labels

print("Cleaned Text:")
for i in range(2):
    print(f"Text: {df['cleaned_text'].iloc[i]}")

vocab_size = len(tokenizer.word_index) + 1
epochs = 10
batch_size = 128

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = getLSTMClassifier(
            X=X,
            vocab_size=vocab_size,
            epochs=epochs,
            batch_size=batch_size
        )
print(model.summary())
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training LSTM classifier ...\n")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping])

print("Evaluate model ...")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {loss:.3f}")
print(f"Validation Accuracy: {accuracy:.3f}")

