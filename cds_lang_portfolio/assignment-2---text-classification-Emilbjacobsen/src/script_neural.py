# system tools
import os
import sys
sys.path.append("..")

# data munging tools
import pandas as pd
#import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
import joblib

# Visualisation
import matplotlib.pyplot as plt

# Argparse
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train-test split")
parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for MLPClassifier")
args = parser.parse_args()

#define path
path_to_csv = os.path.join("..","in","fake_or_real_news.csv")

#creating function to load data and make training split
def load_data():
    filename = path_to_csv
    data = pd.read_csv(filename, index_col=0)

    #taking data out of dataframe
    x = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(x,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=args.test_size,   # create an 80/20 split
                                                        random_state=42) # random state for reproducibility
    return X_train, X_test, y_train, y_test 

#creating function to vectorize data
def vectorize(trainingdata, test_data):
    #create vectorizer
    vectorizer = CountVectorizer(ngram_range = (1,2),    
                                lowercase =  True,       
                                max_df = 0.95,           
                                min_df = 0.05,           
                                max_features = 500)      
                
    # first we fit to the training data
    X_train_feats = vectorizer.fit_transform(trainingdata)

    #... then do it for our test data
    X_test_feats = vectorizer.transform(test_data)

    # get feature names
    feature_names = vectorizer.get_feature_names_out()

    return X_train_feats, X_test_feats

#function to create neural network model and fitting data
def neural_n(xtrainfeats, ytrain, xtestfeats):
    #creating the model
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20, 2), #hvert tal er et hidden layer og antallet er mængde nodes
                           max_iter= args.max_iter, #hvor mange gange man vil have den til at gå over det
                           random_state = 42)
    #fitting data                       
    classifer_1 = classifier.fit(xtrainfeats, ytrain)
    #making predictions
    y_pred = classifier.predict(xtestfeats)

    return y_pred, classifer_1

def get_report(ytest, ypred):
    #making report for 
    classifier_metrics = metrics.classification_report(ytest, ypred)
    
    return classifier_metrics

#function for saving both the report and the model
def saver(report, Classifier_model):
    #defining file path and type
    folder_path = os.path.join("..", "out")
    file_name = "neural_report.txt"
    file_path = os.path.join(folder_path, file_name)
    #saving report
    with open(file_path, "w") as f:
        f.write(report)
    print("reports saved")

    #defining file path and type
    folder_path = os.path.join("..", "models")
    file_name = "neural_model.pkl"
    file_path = os.path.join(folder_path, file_name)
    #using joblib.dump to save model
    joblib.dump(Classifier_model, file_path)

#creating main function to call all previous functions
def main():
    X_train, X_test, y_train, y_test = load_data()

    X_train_feats, X_test_feats = vectorize(X_train, X_test)

    y_pred, classifier_1 = neural_n(X_train_feats, y_train, X_test_feats)

    classifier_metrics = get_report(y_test, y_pred)

    saver(classifier_metrics, classifier_1)

#calling main functions
if __name__=="__main__":
    main()


