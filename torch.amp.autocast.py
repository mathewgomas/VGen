import warnings
import os
import torch
import logging
from argparse import ArgumentParser  # Example import from argparse

# Suppress FutureWarnings from specific libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="xformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="rotary_embedding_torch")

# Initialize logging
logger = logging.getLogger(__name__)

# Example argparse setup (replace with your actual argument parsing code)
parser = ArgumentParser(description='Your script description')
parser.add_argument('--resume', action='store_true', help='Resume training')
parser.add_argument('--work_dir', type=str, default='/path/to/work_dir', help='Path to work directory')
# Add more arguments as needed

# Parse arguments
args = parser.parse_args()

# Set model start epoch
start_epoch = 1
if args.resume:
    model_path = os.path.join(args.work_dir, cfg.MODEL.NAME, 'train', 'latest.pth')
    if os.path.isfile(model_path):
        logger.info(f"=> loaded checkpoint '{model_path}' (epoch {epoch})")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

def load_checkpoint(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"=> loading checkpoint '{model_path}'")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint '{model_path}': {e}")
        return None

# Example usage
model_path = 'path_to_model_checkpoint.pth'
checkpoint = load_checkpoint(model_path)

# Set device
device = torch.device(cfg.GPU.DEVICE)
model.to(device)

# Ensure the rest of your script is properly indented and aligned
