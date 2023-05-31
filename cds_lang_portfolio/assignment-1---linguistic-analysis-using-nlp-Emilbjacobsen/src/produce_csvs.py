import os
import spacy
import pandas as pd
import re
import sys

def prep_and_load():
    directory = os.path.join("..", "in", "USEcorpus")

    nlp = spacy.load("en_core_web_sm")

    return directory, nlp


def analyze(dir, nlp1):
    for folder_name in os.listdir(dir):
        folder_path = os.path.join(dir, folder_name)
        if os.path.isdir(folder_path):
            output = []#creating output folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="latin-1") as file:
                        text = file.read()
                    text_clean = re.sub("<.*?>","",text)#cleaning text
                    doc = nlp1(text_clean) #creating nlp doc

                    entities = [] #creating entities
                    for ent in doc.ents:
                        entities.append(ent.text)

                    adjective_count = 0 #counting word classes and getting frequencies
                    for token in doc:
                        if token.pos_ == "ADJ":
                            adjective_count += 1

                    relative_freq_adj = (adjective_count/len(doc)) * 10000
                    relative_freq_adj = round(relative_freq_adj, 2)


                    verb_count = 0
                    for token in doc:
                        if token.pos_ == "VERB":
                            verb_count += 1

                    relative_freq_verb = (verb_count/len(doc)) * 10000
                    relative_freq_verb = round(relative_freq_verb, 2)

                    noun_count = 0
                    for token in doc:
                        if token.pos_ == "NOUN":
                            noun_count += 1

                    relative_freq_noun = (noun_count/len(doc)) * 10000
                    relative_freq_noun = round(relative_freq_noun, 2)

                    adv_count = 0 #and adverbs
                    for token in doc:
                        if token.pos_ == "ADV":
                            adv_count += 1

                    relative_freq_adv = (adv_count/len(doc)) * 10000
                    relative_freq_adv = round(relative_freq_noun, 2)


                    organisations = [] #finding unique entities

                    for ent in doc.ents:
                        if ent.label_ == "ORG":
                            organisations.append(ent)

                    persons = []
                    for ent in doc.ents:
                        if ent.label_ == "PERSON":
                            persons.append(ent)

                    locations = []
                    for ent in doc.ents:
                        if ent.label_ == "LOCATION":
                            locations.append(ent)

                    set_person = set(persons)
                    set_location = set(locations)
                    set_org = set(organisations)

                    final_person = len(set_person)
                    final_location = len(set_location)
                    final_org = len(set_org)

                    final_data = ((file_name, 
                                    relative_freq_adj, 
                                    relative_freq_verb, 
                                    relative_freq_noun, 
                                    relative_freq_adv, 
                                    final_person, 
                                    final_location, 
                                    final_org))
                    output.append(final_data)
            col_names = ["file_name", 
                                "relative_freq_adj", 
                                "relative_freq_verb", 
                                "relative_freq_noun", 
                                "relative_freq_adv", 
                                "final_person", 
                                "final_location", 
                                "final_org"]
            df = pd.DataFrame(output, columns=col_names)#creating dataframe
            df.to_csv(os.path.join("..","out",f""+folder_name+".csv"))#saving dataframe






def main():
    directory, nlp = prep_and_load()
    analyze(directory, nlp)



if __name__=="__main__":
    main()
