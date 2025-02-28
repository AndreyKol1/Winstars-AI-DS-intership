
import torch
from train_cnn import AnimalClassifier
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

# Defining a parser function, which accepts all required the data from the CLI
def argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data_cnn", required=True, type=str, help="Provide a path to training data") # get the pass to initialize classes names for prediction
    parser.add_argument("--model_dir_cnn", required=True, type=str, help="Provide a path to a model") # path to the trained model
    parser.add_argument("--img_path", required=True, type=str, help="Provide a path to the image") # path to the image to predict on

    
    return parser.parse_args()

# A function for model initialization
def initialize_model(path):
    device = torch.device("cuda")
    model = AnimalClassifier(num_classes=10) # pass through AnimalClassifier class to modify last FC layer
    model = torch.load(f"{path}", weights_only=False, map_location=device) # load model
    model.to(device)

    return model

# A function for testing transformation
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # resize for the resnet18
        transforms.CenterCrop(224), # apply center crop
        transforms.ToTensor(), # make it in tensors for PyTorch computations
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize
    ])
    return transform

def predict(image_path, train_path, model, transform):
    image = Image.open(image_path).convert("RGB")  # open image
    image = transform(image).unsqueeze(0).to("cuda")  # apply transforms and add batch dimension
    
    class_names = [path for path in os.listdir(train_path)] # create a list of animal names
    
    model.eval()
    with torch.inference_mode(): # same as torch.no_grad(), previously set model to evaluation phase, for better performance
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return class_names[predicted_class]

def main():
    args = argument_parser()
    model = initialize_model(args.model_dir_cnn)
    transform = get_transform()
    prediction = predict(args.img_path, args.train_data_cnn, model, transform)
    print(prediction)
    
    
if __name__ == "__main__":
    main()
# python inference_cnn.py --train_data raw-img --model_dir model.pt --img_path sq.jpg