# Solution explanation and comments

## Solution
Our goal was to build a project, which consists of 3 classes for models(Random Forest, NN and CNN) with each one inherits from abstract interface. 

Firstly, I created an abstract class, TrainPredictAbstract, which defines the train and predict methods. Both the FFNNMnistClassifier and CNNMnistClassifier classes inherit from this abstract class to avoid duplicating code. For observation purposes tensorboard was integrated into training loop to conveniently track loss and accuracy of the models.

Secondly, I created a RandomForestMnistClassifier class with train and predict methods, which are working properly, as score of the model indicates.

Thirdly, in classes FFNN and CNN simple architectures of feed-forward neural network and convolutional neural network were built, using some specific features of each architecture. 

Finally, the MnistClassifier class was implemented to allow the user to select the desired model type (Random Forest, NN, or CNN) based on provided parameters.

## Installation  

### CPU-only
After creating and running virtual environment(recommended) go the terminal and execute a command **`pip install -r requirementsCPU.txt`**. This will install all necessary packages to run the project.

### GPU support
After creating and running virtual environment(recommended) go the terminal and execute a command **`pip install -r requirementsGPU.txt`**. This will install all necessary packages to run the project. If installation goes wrong on torch package, try this command **`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`** or visit https://pytorch.org/get-started/locally/ this link to check cuda version compatible with your gpu and then try again running **`pip install -r requirementsGPU.txt`**.
