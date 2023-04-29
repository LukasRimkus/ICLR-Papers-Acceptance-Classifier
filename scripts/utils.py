import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datetime


def get_device() -> torch.device:
    """
    This method checks if GPU is available, if it is, then an object representing it is returned.
    Otherwise, CPU is utilised.  
    """
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')

        print(f'GPU: {torch.cuda.get_device_name(0)}')

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def save_model(output_path: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> None:
    """
    This method is used to save the models in some directory which can be loaded later  
    on for doing other tasks.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def format_time(elapsed: int) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
