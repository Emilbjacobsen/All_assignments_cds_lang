[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10420185&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Text classification benchmarks

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Objective

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

## Some notes

- Saving the classification report to a text file can be a little tricky. You will need to Google this part!
- You might want to challenge yourself to create a third script which vectorizes the data separately, and saves the new feature extracted dataset. That way, you only have to vectorize the data once in total, instead of once per script. Performance boost!

## Additional comments

Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.

For this assignment, you are welcome to submit your code either as a Jupyter Notebook, or as ```.py``` script. If you do not know how to write ```.py``` scripts, don't worry - we're working towards that!

Lastly, you are welcome to edit this README file to contain whatever informatio you like. Remember - documentation is important!



## Contribution

This code was produced in collaboration with others in class. It is based on the notebooks provided by Ross on the same topic.


## Prerequisites
I ran this on ucloud with a 8 cpu machine with 48 gb of memory and on python version 3.9.2



## Methods
This code uses 2 different scripts to train 2 different classifier models that both take vectorized data as input. the data that the model uses as input is the fake or real news dataset. The first script uses a logistic regression classifier to determine which articles are real or fake, the second uses a neural network instead. They both produce classification reports to determine how succesfull they were.

## Data
The data was provided by Ross and the kaggle dataset seems to come in two datasets which wont work with this script, so the dataset i worked with is available here:

https://drive.google.com/file/d/1Wz0irB1LfBdBTBK6bUYSyJLUYHyfELrN/view?usp=sharing




## Running the script
delete gitkeep from datafolder

If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv


Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file 
in terminal with:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the 
same problem, please just run this with assignment2-vis-Emilbjacobsen as working 
directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the 
coder in terminal:

source ./assignment2_lang_env/bin/activate

Then set src as working directory and run:

python3 script_classifier

or

python3 script_neural.py

depending on which model you wish to use.

also use the deactivate command in terminal when done.

## Discussion of results
Both classifiers did relatively well. The logistic regression model has an accuracy score of 0,81 and neural network has 0,88. The logistic regression model is slightly better at finding fake articles than it is at finding real, but the difference is relatively having only a 0.3 difference in f1 score.

logistic regression report:
              precision    recall  f1-score   support

        FAKE       0.79      0.86      0.83       968
        REAL       0.84      0.76      0.80       933

    accuracy                           0.81      1901
   macro avg       0.82      0.81      0.81      1901
weighted avg       0.82      0.81      0.81      1901

nerural network report:
              precision    recall  f1-score   support

        FAKE       0.89      0.87      0.88       628
        REAL       0.88      0.89      0.89       639

    accuracy                           0.88      1267
   macro avg       0.88      0.88      0.88      1267
weighted avg       0.88      0.88      0.88      1267