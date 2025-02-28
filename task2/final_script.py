import argparse
import subprocess

def argument_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--text", type=str, required=True, help="Provide text to predict on")
    parser.add_argument("--image", type=str, required=True, help="Provide an image to predict on")
    parser.add_argument("--ner_script", type=str, default="inference_ner.py", help="Provide a path to a ner inference script")
    parser.add_argument("--image_script", type=str, default="inference_cnn.py", help="Provide a path to a cnn inference script")
    
    return parser.parse_args()

def run_inference(script, input_data):
    result = subprocess.run(["python", script] + input_data, capture_output=True, text=True)
    return result

def bool_output(ner_output, image_output):
    if ner_output.str.lower() == image_output.str.lower():
        return True
    return False

def main():
    args = argument_parser()
    
    ner_output = run_inference(args.ner_script, ["--text", args.text])
    image_output = run_inference(args.image_script, ["--image_script", args.image])
    
    final_result = bool_output(ner_output, image_output)
    
    print(ner_output, image_output, final_result)
    
    # return final_result

if __name__ == "__main__":
    main()