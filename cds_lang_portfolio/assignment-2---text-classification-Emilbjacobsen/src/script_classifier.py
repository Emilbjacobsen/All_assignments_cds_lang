#system tools
import os
import sys
sys.path.append("..")

# data munging tools
import pandas as pd
#import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
import joblib

# Visualisation
import matplotlib.pyplot as plt

# Argparse
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Test size modification')

# Add an argument for text size
parser.add_argument('--test_size', type=float, default=0.2,
                    help='Size of the test split (default: 0.2)')

# Parse the command line arguments
args = parser.parse_args()


#define path
path_to_csv = os.path.join("..","in","fake_or_real_news.csv")

#creating function to load csv and take data out of dataframe to use it and making training split
def load_data():
    filename = path_to_csv
    data = pd.read_csv(filename, index_col=0)

    #taking data out of dataframe
    x = data["text"]
    y = data["label"]
#making training split
    X_train, X_test, y_train, y_test = train_test_split(x,           
                                                        y,          
                                                        test_size=args.test_size,   
                                                        random_state=42) 
    return X_train, X_test, y_train, y_test 

#creating function to vectorize data
def vectorize(trainingdata, test_data):
    #create vectorizer
    vectorizer = CountVectorizer(ngram_range = (1,2),    
                                lowercase =  True,       
                                max_df = 0.95,           
                                min_df = 0.05,           
                                max_features = 100)      
                
    # first we fit to the training data
    X_train_feats = vectorizer.fit_transform(trainingdata)

    #... then do it for our test data
    X_test_feats = vectorizer.transform(test_data)

    # get feature names
    feature_names = vectorizer.get_feature_names_out()

    return X_train_feats, X_test_feats

#function for creating classifier
def classifier_1(X_train_feats, y_train, X_test_feats):
    #creating classifier
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    #getting predictions
    y_pred = classifier.predict(X_test_feats)

    return y_pred, classifier

#functions for getting classifer report
def calssifier_report(ytest, ypred):
    #making the report
    classifier_metrics = metrics.classification_report(ytest, ypred)
    return classifier_metrics

#function for saving both the report and the model
def saver(report, Classifier_model):
    #defining file path and type
    folder_path = os.path.join("..", "out")
    file_name = "classifier_report.txt"
    file_path = os.path.join(folder_path, file_name)
    #saving report
    with open(file_path, "w") as f:
        f.write(report)
    print("reports saved")
    #defining file path and type
    folder_path = os.path.join("..", "models")
    file_name = "classifier_model.pkl"
    file_path = os.path.join(folder_path, file_name)
    #using joblib.dump to save model
    joblib.dump(Classifier_model, file_path)

#creating main function to call all previous functions
def main():

    X_train, X_test, y_train, y_test = load_data()

    X_train_feats, X_test_feats = vectorize(X_train, X_test)

    y_pred, classifier = classifier_1(X_train_feats, y_train, X_test_feats)
    
    classifier_metrics = calssifier_report(y_test, y_pred)

    saver(classifier_metrics, classifier)


#calling main function
if __name__=="__main__":
    main()




