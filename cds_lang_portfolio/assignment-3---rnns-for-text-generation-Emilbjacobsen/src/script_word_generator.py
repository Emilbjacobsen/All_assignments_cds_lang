# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


#import utils
import sys
sys.path.append("..")
import utils.requirement_functions as rf
import joblib

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

path_to_data = os.path.join("..","data")

def load_data(path):
    # Providing path to data dir
    data_dir = path
    # Creating empty list
    all_headlines = []
    # For loop for loading every csv in dir
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            article_df = pd.read_csv(data_dir + "/" + filename)
            # Load only the first hundred comments
            comments = list(article_df["commentBody"].values)[:100]
            all_headlines.extend(comments)
    # Cleaning data
    all_headlines = [h for h in all_headlines if h != "Unknown"]
    
    # Clean text
    corpus = []

    # For loop converting all floats to strings, because the clean_text function was being passed floats
    for x in all_headlines:
        if isinstance(x, float):
            x = str(x)
        cleaned_text = rf.clean_text(x)
        corpus.append(cleaned_text)

    return corpus



def process(corpus):
    tokenizer = Tokenizer()
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    print("tokenized")
    total_words = len(tokenizer.word_index) + 1
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    print("got sequences")
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    print("padded")

    folder_path = os.path.join("..", "out")
    file_name = "tokenizer.joblib"
    file_path = os.path.join(folder_path, file_name)
    #using joblib.dump to save model
    joblib.dump(tokenizer, file_path)

    return inp_sequences, total_words, predictors, label, max_sequence_len




def make_model(inp_sequences, total_words, predictors, label, max_sequence_len):
    #creating and fitting model
    model = rf.create_model(max_sequence_len, total_words)
    history = model.fit(predictors, 
                    label, 
                    epochs=100,
                    batch_size=128, 
                    verbose=1)
    
    # Load the tokenizer from the saved joblib file
    tokenizer_path = os.path.join("..","out","tokenizer.joblib" )
    tokenizer = joblib.load(tokenizer_path)
    #saving model
    folder_path = os.path.join("..", "model")
    file_name = "text_generation_model.h5"
    file_path = os.path.join(folder_path, file_name)
   
    model.save(file_path)


def main():
    corpus = load_data(path_to_data)
    print("data loaded")
    inp_sequences, total_words, predictors, label, max_sequence_len = process(corpus)
    print("process complete")
    make_model(inp_sequences, total_words, predictors, label, max_sequence_len)
    print("model made")


if __name__=="__main__":
    main()
    