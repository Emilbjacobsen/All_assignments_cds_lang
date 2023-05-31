#importing packages
from transformers import pipeline
import pandas as pd
import numpy as np
import os

#path to the dataset
path_to_csv = os.path.join("..", "data", "fake_or_real_news.csv")

#Making function for loading data
def load_data(path_to_csv):
    filename = path_to_csv
    df = pd.read_csv(filename)


    #creating new datasets with only the real or fake articles
    real_headlines_df = df[df['label'] == 'REAL']

    fake_headlines_df = df[df['label'] == 'FAKE']

    #assigning the title collumn to object
    real_headlines = real_headlines_df["title"]
    fake_headlines = fake_headlines_df["title"]


    return real_headlines, fake_headlines

#Function for processing data
def process(headline_list):
    #loading classifier model
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=False)

    #making empty list
    outputs = []
    #for loop putting headlines into list
    for headline in headline_list:
        outputs.append(classifier(headline))

    #renaming list to variable name
    headline_list = outputs

    #loop, flattening the list and turning it into a list of dicts
    headline_list1 = [item for sublist in headline_list for item in sublist]
    return headline_list1

#Function for putting the lists into 1 dataframe so we can make visualisations
def make_df(emote_realrows, emote_fakerows):



    #making seperate df's for real, fake and all articles
    fake_news_df = pd.DataFrame(emote_realrows, columns=['label', 'score'])
  
    real_news_df = pd.DataFrame(emote_fakerows, columns=['label', 'score'])
  
    #combining previos pd's to one to get all articles
    all_news_df = pd.concat([real_news_df, fake_news_df], axis=0)


    return fake_news_df, real_news_df, all_news_df



#function for saving
def saver(fake_news_df, real_news_df, all_news_df):
    all_news_df.to_csv(os.path.join("..", "table", "all_news_df.csv"))#saving dataframe
    real_news_df.to_csv(os.path.join("..", "table", "real_news_df.csv"))
    fake_news_df.to_csv(os.path.join("..", "table", "fake_news_df.csv"))



#Main function
def main():
   real_headlines, fake_headlines = load_data(path_to_csv)
   emote_realrows = process(real_headlines)
   emote_fakerows = process(fake_headlines)
   fake_news_df, real_news_df, all_news_df = make_df(emote_fakerows, emote_realrows)
   saver(fake_news_df, real_news_df, all_news_df)




if __name__=="__main__":
    main()