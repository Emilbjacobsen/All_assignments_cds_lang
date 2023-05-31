import csv
import os
import pandas as pd
import matplotlib.pyplot as plt



csv_dir = os.path.join("..", "table")



def get_viz(file, filename):
    
    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # group the data by label and count the number of entries for each label
    label_counts = df.groupby('label').count()['score']

    # create a bar plot
    label_counts.plot(kind='bar')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Entries')
    plt.title('Distribution of Emotions')

    plt.xticks(rotation=0)

    

    plt.savefig(os.path.join("..", "visualisations", f""+filename+".png"))
    plt.close()
 
def get_viz2(file, filename):
    
    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Group the DataFrame by emotion and calculate the average score for each emotion
    grouped_df = df.groupby("label")["score"].mean().reset_index()

    # Create a bar chart using Matplotlib to visualize the average score for each emotion
    plt.bar(grouped_df["label"], grouped_df["score"])

    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Average Emotion Score')

    plt.savefig(os.path.join("..", "visualisations", f""+filename+"_score.png"))
    plt.close()


for csv in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, csv)
    get_viz(file_path, csv)
    get_viz2(file_path, csv)

