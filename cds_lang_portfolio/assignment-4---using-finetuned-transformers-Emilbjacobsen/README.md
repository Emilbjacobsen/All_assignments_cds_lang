# Assignment 4 - Using finetuned transformers via HuggingFace

## Github link and contributions
Here is the github repo:

https://github.com/AU-CDS/assignment-4---using-finetuned-transformers-Emilbjacobsen

this assignment was made by me alone.

## Assignment description
For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines

## Contribution

This code was produced in collaboration with others in class and is based on code ross provided in class in his notebooks.


## Methods
These two scripts are based on the real or fake news dataset. The first script downloads a pretrained model from huggingface, it uses this model for sentiment analysis on the real fake news dataset.  

## Hardware
This code was run on ucloud with a 16 with 96 gb memory.


## Data
The data was provided by Ross and the kaggle dataset seems to come in two datasets which wont work with this script, so the dataset i worked with is available here:

https://drive.google.com/file/d/1Wz0irB1LfBdBTBK6bUYSyJLUYHyfELrN/view?usp=sharing



## Running the script
Delete gitkeep from datafolder

If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv


Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file 
in terminal with:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the 
same problem, please just run this with assignment4-vis-Emilbjacobsen as working 
directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the 
coder in terminal:

source ./assignment4_lang_env/bin/activate

Then set src as working directory and run:

python3 pipeline.py

and then

python3 visualisations_OG.py



also use the deactivate command in terminal when done.

## Discussion of results
Based on the visualisations created by the script, there are no significant differences in the sentiment distribution of the real and fake news. The model is however slightly more secure in its predictions regarding the sentiments of the fake news, as can be seen by its scores are slightly higher.
