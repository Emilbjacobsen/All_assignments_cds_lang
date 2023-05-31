import argparse
import string, os 
import sys
import joblib
import numpy
sys.path.append("..")
import utils.requirement_functions as rf
import tensorflow as tf
from tensorflow.keras.models import load_model


# Define the command-line arguments
parser = argparse.ArgumentParser(description='Generate text from a user-suggested prompt')

parser.add_argument('--prompt', metavar='prompt', type=str, help='Input prompt for text generation')
parser.add_argument('--sentence_length', metavar='sentence_length', type=int, help='Length of generated sentences', default = 10)

# Parse the command-line arguments
args = parser.parse_args()


# Paths to model and tokenizer
toke_path = os.path.join("..","out","tokenizer.joblib" )
model_path1 = os.path.join("..","model","text_generation_model.h5" )
# Load the saved model and tokenizer
tokenizer = joblib.load(toke_path)
model = load_model(model_path1)


# Generate the text based on the user-suggested prompt
generated_text = rf.generate_text(args.prompt, args.sentence_length, model, 272)

# Print the generated text
print(generated_text)