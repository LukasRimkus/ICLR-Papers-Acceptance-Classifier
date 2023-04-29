import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.data_preprocessing import create_dataloader
from scripts.evaluation_metrics import display_model_performance_metrics, return_metrics
from scripts.utils import save_model


def make_predictions(testing_dataloader: DataLoader, device: torch.device, model: AutoModelForSequenceClassification) -> tuple:
    """
    This method makes sentiment predictions for the whole dataset of documents. Dataloader
    is iterated through to predict the results of a batch of documents to not fill the whole 
    available memory.  
    """
    # Put model in evaluation mode
    model.eval()

    predictions, true_labels, probabilities = [], [], []
    
    for batch in testing_dataloader:
        logits, label_ids, probs = batch_predict(batch, device, model)
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        probabilities.extend(probs)
    
    probabilities = np.array(probabilities)

    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Display the results for each class
    display_model_performance_metrics(flat_true_labels, flat_predictions)
    accuracy, precision, recall, f1_value = return_metrics(flat_true_labels, flat_predictions)
    
    return accuracy, precision, recall, f1_value, probabilities

    
def batch_predict(batch: torch.Tensor, device: torch.device, model: AutoModelForSequenceClassification) -> tuple:
    """
    Generate and return model predictions for the batch of tweets.
    """
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = model(b_input_ids, 
                     attention_mask=b_input_mask,
                     return_dict=True)

    logits = result.logits
    
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    return logits, label_ids, probs


def evaluate_tested_model(model: AutoModelForSequenceClassification, model_name: np.ndarray, tokenizer: AutoTokenizer,
                                testing_documents: np.ndarray, testing_labels: np.ndarray, batch_size: int, device: torch.device, 
                                MAX_TOKENS_NUMBER: int, model_path: str=None) -> None:
    """
    This method produces testing results for each specified model architecture. 
    """
    print(f"============ Transformer {model_name} evaluation metrics ============")
    
    testing_dataloader = create_dataloader(testing_documents, testing_labels, SequentialSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)    

    # Obtain the testing results for the model architecture
    accuracy, _, _, _, _ = make_predictions(testing_dataloader, device, model)

    # Optionally store the model with its accuracy which can be used for papers decision inference
    if model_path:
        output_path = f"{model_path}/{model_name.replace('/', '-')}_acc_{accuracy:.4f}"
        save_model(output_path, model, tokenizer)
