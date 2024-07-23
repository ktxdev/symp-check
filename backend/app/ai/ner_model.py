import os
import spacy
import random
import pandas as pd

from spacy.training import Example
from spacy.util import minibatch, compounding

def preprocess_data(df):
    train_data = []

    for _, row in df.iterrows():
        symptom = row['symptom']
        sentence = row['sentence']

        start_index = sentence.index(symptom)
        end_index = start_index + len(symptom)

        train_data.append((sentence, {"entities": [(start_index, end_index, 'SYMPTOM')]}))

    return train_data

def train_ner_model(data):
    # Get the pipeline component
    ner = nlp.get_pipe("ner")

    # Add the new entity label
    ner.add_label("SYMPTOM")

    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.create_optimizer()

        for i in range(30):  # number of iterations
            random.shuffle(data)

            losses = {}
            batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                texts, annotations = zip(*batch)
                example = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
                nlp.update(example, drop=0.5, losses=losses, sgd=optimizer)
                
            print("Losses", losses)


if __name__ == "__main__":

    df = pd.read_csv("https://raw.githubusercontent.com/ktxdev/symp-check/main/backend/data/processed/symptoms_sentences.csv")

    data = preprocess_data(df)

    nlp = spacy.load("en_core_web_sm")

    train_ner_model(data)

    # Save model
    os.makedirs('data/models', exist_ok=True)
    nlp.to_disk("data/models/er_symptom_model")