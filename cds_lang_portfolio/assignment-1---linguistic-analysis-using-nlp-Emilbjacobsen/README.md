
# Language analytics assignment 1

## Assignment task

This assignment concerns using ```spaCy``` to extract linguistic information from a corpus of texts.

The corpus is an interesting one: *The Uppsala Student English Corpus (USE)*. All of the data is included in the folder called ```in``` but you can access more documentation via [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

For this exercise, you should write some code which does the following:

- Loop over each text file in the folder called ```in```
- Extract the following information:
    - Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 10,000 words
    - Total number of *unique* PER, LOC, ORGS
- For each sub-folder (a1, a2, a3, ...) save a table which shows the following information:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|file1.txt|---|---|---|---|---|---|---|
|file2.txt|---|---|---|---|---|---|---|
|etc|---|---|---|---|---|---|---|

## Objective

This assignment is designed to test that you can:

1. Work with multiple input data arranged hierarchically in folders;
2. Use ```spaCy``` to extract linguistic information from text data;
3. Save those results in a clear way which can be shared or used for future analysis

## Some notes

- The data is arranged in various subfolders related to their content (see the [README](in/README.md) for more info). You'll need to think a little bit about how to do this. You should be able do it using a combination of things we've already looked at, such as ```os.listdir()```, ```os.path.join()```, and for loops.
- The text files contain some extra information that such as document ID and other metadata that occurs between pointed brackets ```<>```. Make sure to remove these as part of your preprocessing steps!
- There are 14 subfolders (a1, a2, a3, etc), so when completed the folder ```out``` should have 14 CSV files.

## Additional comments

Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.

For this assignment, you are welcome to submit your code either as a Jupyter Notebook, or as ```.py``` script. If you do not know how to write ```.py``` scripts, don't worry - we're working towards that!

Lastly, you are welcome to edit this README file to contain whatever informatio you like. Remember - documentation is important!

## Contribution

This code was produced in collaboration with others in class.


## Prerequisites
I ran this on ucloud with a 8 cpu machine with 48 gb of memory and on python version 3.9.2



## Methods
This code uses the spacy module for a loop to iterate over a large amount of textfiles, finding word classes and different unique entities such as people, locations and organisations

## Data
The data can be found here:

https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457

Download the USEcorpus.zip file.


## Running the script
delete gitkeep from datafolder

If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv


Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file 
in terminal with:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the 
same problem, please just run this with assignment1-vis-Emilbjacobsen as working 
directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the 
coder in terminal:

source ./assignment1_lang_env/bin/activate

Unzip the data to the in folder and then set src as working directory and run:

python3 produce_csvs.py

also use the deactivate command in terminal when done.

## Discussion of results
The code works as intended and produces the csv's as intended. This script could potentially be used for text analysis of large corpuses of text data. The code is in functions, but because the script is 1 big for loop the number of functions that could be made was limited.