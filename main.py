import json
import argparse
import os
import torch

# Set OpenMP environment variable to suppress multiple initialization warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set environment variable to avoid potential multi-process issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """Main training entry point"""
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()  # Clear GPU cache
    else:
        print("Warning: No GPU detected, will use CPU for training (will be very slow)")
    
    # Start training based on model selection
    model_name = args.get("model_name", "_")
    if model_name == "_":
        from trainer_morphology import train
        train(args)
    else:
        from trainer import train
        train(args)

def load_json(settings_path):
    """Load configuration from JSON file"""
    print(f"Loading configuration file: {settings_path}")
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Stage-Aware Class-Incremental Learning')
    
    parser.add_argument('--config', type=str, default='./exps/StageCIL.json',
                        help='Configuration file path')
    parser.add_argument('--name', type=str, default='',
                        help='Experiment name')
    parser.add_argument('--model_name', type=str, default='_,
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='_',
                        help='Dataset name')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--lambda_rehearsal', type=float, default=0.1,
                        help='Prototype rehearsal loss weight')
    parser.add_argument('--alpha_noise', type=float, default=0.1,
                        help='Prototype enhancement Gaussian noise standard deviation')
    
    return parser

if __name__ == '__main__':
    main()
