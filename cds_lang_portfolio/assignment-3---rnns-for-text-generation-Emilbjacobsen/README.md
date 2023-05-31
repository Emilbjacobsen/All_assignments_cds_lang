
# Assignment 3 - Language modelling and text generation using RNNs

Text generation is hot news right now!

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt


## Some tips

One big thing to be aware of - unlike the classroom notebook, this assignment is working on the *Comments*, not the articles. So two things to consider:

1) The Comments data might be structured differently to the Articles data. You'll need to investigate that;
2) There are considerably more Comments than articles - plan ahead for model training!

## Additional pointers

- Make sure not to try to push the data to Github!
- *Do* include the saved models that you output
- Make sure to structure your repository appropriately
  - Include a readme explaining relevant info
    - E.g where does the data come from?
    - How do I run the code?
- Make sure to include a requirements file, etc...

## Contribution

This code was produced in collaboration with others in class and is based on code ross provided in class. It is also based on some of the notebooks Ross provided for the class, specifically the notebook on text generation.


## Methods
This project is made up off 2 scripts. The first script process multiple CSVs containing comments and trains a text generation model on a subset of this data. This is because the machine available to me was not good enough to use the entire dataset. This has a significant influence on the result. The script then saves the model. The second script is a prompter script that loads the model and lets the user give the model prompt via argparse from which the model can generate text from.

## Hardware
this code was run on ucloud with a 32 with 192 gb memory.

## Data
The data can be found here:
https://www.kaggle.com/datasets/aashita/nyt-comments




## Running the script
delete gitkeep from datafolder

If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv


Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file 
in terminal with:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the 
same problem, please just run this with assignment3-vis-Emilbjacobsen as working 
directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the 
coder in terminal:

source ./assignment3_lang_env/bin/activate

Unzip the data to the data folder.

Then set src as working directory and run:

python3 script_word_generator.py

and then

python3 script_prompter.py --prompt <"your prompt here">
(example)
python3 script_prompter.py --prompt "the united states of america"

depending on which classifier you wish to train.

also use the deactivate command in terminal when done.

## Discussion of results
The model was trained only 100 comments from each csv since it would crash when run on larger parts of the dataset and it still took 6 hours to train on a 32 cpu.. The prompt generator works as intended put its outputs doesn't really make sense most of the time, this is probably due to relatively small amount of data it was trained on. Given a larger amount of data the model would probably perform significantly better.
