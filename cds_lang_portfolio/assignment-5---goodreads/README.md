# Assignment 5 -Self assigned project – Genre classifier 

## Description

This project works with the goodreads best books dataset, and it seeks to determine whether or not, a books genre can be determined by the book’s blurb (a short description of the book, usually found on the back of the book). To accomplish this goal, I will train a multilabel bidirectional LSTM model text classifier to assign genres to each blurb.
To determine the effectiveness of the classifier the script produces a classification report and an accuracy and loss curve.


## Contribution

This code was produced by me with significant inspiration from these following kaggle notebooks.

https://www.kaggle.com/code/kkhandekar/bidirectional-lstm-poem-classification

https://www.kaggle.com/code/leireher/book-genres




## Methods
This script takes the goodreads dataset and cleans it, it does so by converting everything to lower case and removing stopwords. It then trains a bidirectional LSTM model to classify the blurbs. The model is a multilabel classifier, meaning that each blurb can have multiple labels. though there are many more genres in the data i chose to limit genre labels to the most often occurring 20 genres, excluding the audiobook label. It does not use word embeddings because i found that though they did make the script run much faster, they also significantly reduced performance, so the model trains its own. The model itself is made up of two bidirectional LSTM layers and 3 fully connected layers. Less complicated models showed to not make much improvement beyond the first epoch. The script is also run through a virtual environment to make sure there are no package conflicts.

## Prerequisites
This code was run on ucloud with a 32 with 192 gb memory and on python version 3.9.2.


## Data
The data can be found here:

https://www.kaggle.com/datasets/arnabchaki/goodreads-best-books-ever



## Running the script
Delete gitkeep from datafolder

If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv

Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file 
in terminal with:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the 
same problem, please just run this with assignment5-vis-Emilbjacobsen as working 
directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the 
coder in terminal:

source ./assignment5_lang_env/bin/activate

Then unzip the data into the in folder and set src as working directory and run:

python3 train_model.py

also use the deactivate command in terminal when done.


## Discussion of results
The accuracy of this model is 0.50. There is a large difference from genre to genre, in how effective the model is at assigning labels, with the highest f1 score being 0.79 for fiction and the lowest being 0.04 for humor, the rest of the labels have f1 scores are in-between though no other labels go below 0.30. The model does seem to be more successful in the genres which have more examples, showing that the model might benefit from an expansion of the dataset.

                    precision    recall  f1-score   support

             Adult       0.50      0.25      0.33      1645
         Adventure       0.51      0.36      0.43      1321
         Childrens       0.59      0.37      0.45      1022
          Classics       0.61      0.23      0.34      1377
      Contemporary       0.71      0.49      0.58      2117
           Fantasy       0.75      0.75      0.75      3037
           Fiction       0.78      0.80      0.79      6309
        Historical       0.60      0.43      0.50      1330
Historical Fiction       0.74      0.41      0.53      1582
             Humor       0.33      0.02      0.04       857
        Literature       0.59      0.30      0.40      1188
             Magic       0.52      0.53      0.52       892
           Mystery       0.72      0.38      0.50      1536
        Nonfiction       0.71      0.67      0.69      1636
            Novels       0.52      0.30      0.38      1582
        Paranormal       0.72      0.61      0.66      1244
           Romance       0.78      0.62      0.69      3118
   Science Fiction       0.53      0.30      0.38      1068
          Thriller       0.66      0.40      0.50       924
       Young Adult       0.63      0.53      0.58      2375

         micro avg       0.69      0.52      0.59     36160
         macro avg       0.62      0.44      0.50     36160
      weighted avg       0.67      0.52      0.57     36160
       samples avg       0.59      0.47      0.50     36160

Because of large training times, the classifier i was only able to run the model for 3 epochs (for some reason the accuracy and loss curves only show the first two epochs), each epoch lasting from about 2 hours. The accuracy and loss curves show that this model would benefit from further training, though probably not much more. The long training time could be reduced by using embeddings, but I have not found any that did not reduce performance to the point where removing them wasn't the better option.

