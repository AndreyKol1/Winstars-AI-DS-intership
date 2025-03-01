import argparse
import subprocess

# Final argument parser, which contains everything needed for predictions
def argument_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--text", type=str, required=True, help="Provide text to predict on") # text to predict
    parser.add_argument("--model_dir_ner", type=str, required=True, help="Provide a path to a ner model") # ner model directory

    parser.add_argument("--train_data_cnn", required=True, type=str, help="Provide a path to training data") # get the pass to initialize classes names for prediction
    parser.add_argument("--model_dir_cnn", required=True, type=str, help="Provide a path to a model") # path to the trained model
    parser.add_argument("--img_path", required=True, type=str, help="Provide a path to the image") # path to the image to predict on


    parser.add_argument("--ner_script", type=str, default="inference_ner.py", help="Provide a path to a ner inference script") # optional, if renamed or moved to a folder
    parser.add_argument("--image_script", type=str, default="inference_cnn.py", help="Provide a path to a cnn inference script") # optional, if renamed or moved to a folder

    return parser.parse_args()

def run_inference(script, input_data):
    result = subprocess.run(["python", script] + input_data, capture_output=True, text=True) # runs provided script with data required for inference scripts
    return result.stdout.strip().lower() # return lowered output 

# Comparison function of outputs
def bool_output(ner_output, image_output):
    return ner_output == image_output

def main():
    args = argument_parser()
    
    ner_output = run_inference(args.ner_script,  ["--model_dir_ner", args.model_dir_ner, "--text", args.text]) # required params for ner model prediction
    image_output = run_inference(args.image_script, ["--train_data_cnn", args.train_data_cnn, "--model_dir_cnn", args.model_dir_cnn, "--img_path", args.img_path]) # required params for cnn model prediction
    
    final_result = bool_output(ner_output, image_output)
    
    print(f"NER output: {ner_output}")
    print(f"Image output: {image_output}")
    print(f"Final result: {final_result}")
    

if __name__ == "__main__":
    main()