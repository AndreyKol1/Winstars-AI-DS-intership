import spacy 
import random
from spacy.util import minibatch
from spacy.training.example import Example
import argparse
import json

# defining a parser function, which accepts all the data from the CLI
def argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type=str, required=True, help="Provide a path to a json file with training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide a path, where you will store a model")
    
    parser.add_argument("--epochs", type=int, default=20, help='Provide number of epochs') # optional parameters for epochs
    parser.add_argument("--learning_rate", type=int, default=1e-5, help='Provide a learning rate') # optional parameter for learning rate
    
    
    return parser.parse_args()

# A function for loading the data
def dataloader(path):
    with open(path, "r") as f:
        return json.load(f)
    
# A function for processing the data, converting it to a spacy format
def preprocessData(data, nlp):
    training_examples = [] # list for storing preprocessed data
    
    for text, annotations in data:
        doc = nlp.make_doc(text) # make_doc tokenizes the text by splitting each word in the sentence
        example = Example.from_dict(doc, annotations) # Creating an Example object, which tells our model which tokens corresponds to name entities
        training_examples.append(example)
        
    return training_examples

# A function for training a model
def train(training_data, output_dir, epochs=20, lr=1e-5):
    nlp = spacy.load("en_core_web_trf") # loading a transformer

    if "transformer" not in nlp.pipe_names: 
        nlp.add_pipe("transformer", first=True) # adding transformer pipeline if not exists
    
    if "ner" not in nlp.pipe_names: # if ner not exist it adds it to a pipeline, otherwise retrieves it
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    for _, annotations in training_data: # loop for adding labels 
        for ent in annotations['entities']:
            if ent[2] not in ner.labels:
                ner.add_label(ent[2])
                
    data = preprocessData(training_data, nlp) # receiving data in a proper format for training
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ['ner', 'transformer']] # getting other pipes if exists
    with nlp.disable_pipes(*other_pipes): # disabling all pipelines except transformer and ner
        optimizer = nlp.resume_training() # initialize optimizer
        optimizer.learn_rate = lr
        
        # training loop
        for epoch in range(epochs):
            random.shuffle(data)
            losses = {}
            batches = minibatch(data, size=4)
            for batch in batches:
                nlp.update(batch, drop=0.3, losses=losses)
            print(f'Epoch {epoch + 1}, Loss: {losses["ner"]}')
    
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
 
# A wrapper function   
def main():
    args = argument_parser()
    train_data = dataloader(args.train_data)
    spacy.require_gpu()
    train(train_data, args.output_dir, args.epochs)
    

if __name__ == "__main__":
    main()
        


