# Solution explanation and comments

## Solution

In second task our goal was to build a pipeline that includes two models to check whether animal mentioned in user-given sentence matches the animal on the photo given to the other model.

For the NLP part, I used **`en_core_web_trf`**, based on RoBERTa base model, transformer for extracting animals from the sentences. The dataset consists of 260-300 simple and complex sentences mentioning animal with annotations for animal position in the sentence. The mix of simple and complex sentences helped to prevent overfit, despite the small dataset(first few epochs loss still fluctuates, but then consistently goes down). This method of learning helped the model to extract animal names from the complex sentences with very good accuracy and easily from simple sentences. 

For the CV part, I used transfer-learning technique in combination with data-augmentation, which helped my model to score more than 96% validation accuracy only for 3 epochs. The resnet18 pre-trained model was used. I included the ability to play with epochs and learning rate, with combination of tracking loss and accuracy on tensorboard, I think model can score even more. Since, the model have performed really well for 3 epochs I decided not to include fine-tuning technique because it would have been just a waste of time. 

## Commands

### NER model training

**`python train_ner.py --train_data ner_dataset.json --output_dir ner_model`** a command for training NER model. The model will be saved to ner_model folder. If you made some name changes in files or folders, don't forget to adjust the command.

### CNN model training

**`python train_cnn.py --train_data raw-image --output_dir cnn_model`** a command for training CNN model. The model will be saved to cnn_model folder. If you made some name changes in files or folders, don't forget to adjust the command.

### Get prediction

**`python final_script.py --text "Your sentence" --model_dir_ner ner_model --train_data_cnn raw-img --model_dir_cnn cnn_model/model.pt --img_path yourimage.jpg`** a command to get results of comparison. If you made some name changes in files or folders, don't forget to adjust the command.


## Installation

This project is supported only with GPU graphic card. To set up a this project you need to create virtual environment with python version of 3.11. if you are using conda: **`conda create -n envname python=3.11`**. If using virtual environment: **`python3.11 -m envname .env`**
After activation of virtual environment execute a command **`pip install -r requirements.txt`**. This will install all necessary packages to your environment. Them you can use commands written above to train and test the models.  