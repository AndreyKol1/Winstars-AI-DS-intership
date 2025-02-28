import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os

# Defining a parser function, which accepts all required the data from the CLI
def argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type=str, required=True, help="Provide a path to a json file with training data") # path to a dataset
    parser.add_argument("--output_dir", type=str, required=True, help="Provide a path, where you will store a model") # path for saving a model
    
    parser.add_argument("--epochs", type=int, default=3, help='Provide number of epochs') # optional parameters for epochs
    
    
    return parser.parse_args()

# A function to apply data-augmentation and image transformation
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # resize to 256, because resnet18 requires this resolution
        transforms.CenterCrop(224), # cropping to get better center view
        transforms.RandomHorizontalFlip(), # flipping an image with probability=0.5
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1), # add color parameters
        transforms.ToTensor(), # make it in tensors for PyTorch computations
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # best suitable numbers for resnet18
    ])
    return transform

# Creating a custom dataset 
class AnimalDataset:
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform) # since the dataset has appropriate format(raw_img/animal_name/xxx.jpg..) we can use ImageFolder
        
    def get_dataloaders(self):
        train_size = int(0.8 * len(self.dataset)) # 80% for training
        valid_size = len(self.dataset) - train_size # 20% for testing
        
        train_data, valid_data = random_split(self.dataset, [train_size, valid_size]) # random split of the data
        
        train_dl = DataLoader(train_data, batch_size=128, shuffle=True) # training dataset
        valid_dl = DataLoader(valid_data, batch_size=128, shuffle=False) # validation dataset
        
        return train_dl, valid_dl

# Creating a classifier with a pre-trained resnet18 model for transfer learning   
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__() # inherit from nn.Module to have access to modules of this class
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # setting up a model with default weight
        in_features = self.model.fc.in_features # getting input features of last fully connected layer
        self.model.fc = nn.Linear(in_features, num_classes) # modifying last fully connected layer for our conditions(10 classes)
        
    def forward(self, x):
        return self.model(x)
    
# Creating a training loop
def train(model, train_dl, valid_dl, output_dir, epochs=3, lr=1e-4):
        device = torch.device("cuda") # since there is no sense to train it on cpu we will only use gpu
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # setting up Adam optimizer
        loss_fn = nn.CrossEntropyLoss() # setting up CrossEntropyLoss for multi-class prediction
    
        scaler = GradScaler() # initializing GradScaler for mixed precisions(using float16 where possible) to reduce memory usage, faster computations(works only with GPU). 
        torch.backends.cudnn.benchmark = True # by setting benchmark to True PyTorch will find the fastest algorithm on operations like convolutions for your hardware
        
        writer = SummaryWriter() # initializing SummaryWriter instance to enable tensorboard to track train and validation losses and accuracies

        for epoch in range(epochs):
            model.train() # setting model to train phase
            
            # variables to track model performance on training data
            total_samples_train = 0
            total_loss_train = 0
            total_correct_train = 0
            
            for x_batch, y_batch in train_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device) # transfering x_batch, y_batch to cuda
                optimizer.zero_grad() # clearing gradients to avoid their summation

                with autocast(device_type="cuda"): # enabling mixed precisions
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)

                scaler.scale(loss).backward() # scaling loss before backpropagation serves stable performance by preventing gradient underflow in float16
                scaler.step(optimizer) # unscales gradients 
                scaler.update() # update weights

                total_loss_train += loss.item() * y_batch.size(0) # here multiplying by y_batch.size(0) normalizes the loss
                total_correct_train += (torch.argmax(pred, dim=1) == y_batch).sum().item() # sum of correct labels
                total_samples_train += y_batch.size(0) # counter for size
        
            train_loss = total_loss_train / len(train_dl.dataset) # epoch train loss
            train_accuracy = total_correct_train / total_samples_train # epoch train accuracy


            model.eval() # setting model to evaluation phase

            # variables to track model performance on validation data
            total_correct_valid = 0
            total_samples_valid = 0
            total_loss_valid = 0

            with torch.inference_mode(): # disabling gradients calculation to measure the model performance during epoch
                for x_batch, y_batch in valid_dl:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device) # transfering x_batch, y_batch to cuda

                    with autocast(device_type="cuda"): # enabling mixed precisions, float16, where possible
                        pred = model(x_batch)
                        loss = loss_fn(pred, y_batch)

                    total_loss_valid += loss.item() * y_batch.size(0) 
                    total_correct_valid += (torch.argmax(pred, dim=1) == y_batch).sum().item()
                    total_samples_valid += y_batch.size(0)

            valid_loss = total_loss_valid / len(valid_dl.dataset)
            valid_accuracy = total_correct_valid / total_samples_valid


            # writing losses and accuracies to folders to track them in tensorboard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", valid_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Valid", valid_accuracy, epoch)

            print(f'Epoch {epoch + 1}, train acc: {train_accuracy:.4f}, valid acc: {valid_accuracy:.4f}')

        writer.close()
        
        os.makedirs(output_dir, exist_ok=True) # creating a directory for our model
        torch.save(model, os.path.join(output_dir, "model.pt")) # saving model to newly created directory
        
        
def main():
    args = argument_parser()
    transform = get_transforms()
    dataset = AnimalDataset(args.train_data, transform)
    train_dl, valid_dl = dataset.get_dataloaders()
    model = AnimalClassifier(num_classes=10)
    train(model, train_dl, valid_dl, args.output_dir)
    
if __name__ == "__main__":
    main()
    
        
    
    
    
        
        