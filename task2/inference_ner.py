import spacy 
import argparse

def argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir_ner", type=str, required=True, help="Provide a path to a trained model")
    parser.add_argument("--text", type=str, required=True, help="Enter a sentence to get prediction")
    
    return parser.parse_args()

# A function for model loading wia received path
def load_model(path):
    return spacy.load(path)


# A function for making prediction
def predict(nlp, text):
    doc = nlp(text) # passing input text through the pipeline 
    entities = [(ent.text, ent.label_) for ent in doc.ents] # extracting text and labels from text
    return entities[0][0] if entities else "No Entity" # Extacting entity

# A wrapper function 
def main():
    args = argument_parser()
    model = load_model(args.model_dir_ner)
    prediction = predict(model, args.text)
    print(prediction)
    
if __name__ == "__main__":
    main()

# python inference_ner.py --model_dir out/ --text "An elephant and a cat are resting under a tree in the photo."
