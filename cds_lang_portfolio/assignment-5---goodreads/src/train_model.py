# Packages
import pandas as pd
import ast
import os
import re
import pickle
import numpy as np
import ast
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Split, vectorizer, multilabel binarizer and stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report

# Model parameters, tokenizer and pad_sequences functions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Plot libraries
import matplotlib.pyplot as plt

# Function for removing stopwords and oterwise cleaning the data
def clean(text):
     # Replace floats with strings
    if isinstance(text, float):
        text = str(text)

    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub("[^a-z0-9<>]", ' ', text)

    # Filtering stopwords
    filtered_text = " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)


    return filtered_text


def data_prep():
    data = pd.read_csv('../in/goodreads.csv')

    data = data[['title','description','genres']]

    data['genres'] = data['genres'].apply(ast.literal_eval)

    word_list = [word for sublist in data['genres'] for word in sublist if word != "Audiobook"]

    # Count the occurrences of each word
    word_counts = Counter(word_list)

    # Get the 50 most common unique values
    top_20_genres = [value for value, count in word_counts.most_common(20)]

    #Filter the 'genres' column by keeping only the values in the top 20 genres
    data['genres'] = data['genres'].apply(lambda x: [genre for genre in x if genre in top_20_genres])

    #applying clean function to description col
    data['clean_desc'] = data['description'].apply(clean)


    return data

def train_model(data):
    
    # Preprocessing
    X = data['clean_desc'].values
    y = data['genres'].values  # Convert string representation of list to actual list

    # Convert labels to binary format
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad sequences to have the same length
    max_length = max(len(seq) for seq in X_train)
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    # Build the model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 150

    num_classes = y.shape[1]  # Number of unique genres

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))


    return X_test, y_test, model, history, mlb

def get_rep(X_test, y_test, model, mlb):
    # Classification report
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=mlb.classes_)

    #defining file path and type
    folder_path = os.path.join("..", "out")
    file_name = "classification_report.txt"
    file_path = os.path.join(folder_path, file_name)
    #saving report
    with open(file_path, "w") as f:
        f.write(report)
    print("reports saved")

def get_curves(history):
    # Accuracy curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join("..", "out", "acc_curve.png"))
    plt.close()

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("..", "out", "loss_curve.png"))
    plt.close()

def main():
    # Getting and cleaning data
    data = data_prep()

    # Training model
    X_test, y_test, model, history, mlb = train_model(data)

    # Getting classification report
    get_rep(X_test, y_test, model, mlb)

    # Getting accuracy and loss curves
    get_curves(history)


# Calling main function
if __name__=="__main__":
    main()



