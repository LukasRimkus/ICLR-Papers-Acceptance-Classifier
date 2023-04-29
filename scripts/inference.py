from scripts.data_reading import parse_paper_from_json, parse_paper_from_pdf_document

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, Sampler, SequentialSampler

import numpy as np


def create_dataloader_for_inference(documents: np.ndarray, sampler: Sampler, batch_size: int, tokenizer: AutoTokenizer, MAX_TOKENS_NUMBER: int) -> DataLoader:
    """
    This method creates a dataloader object given parameters max number of tokens, 
    type of the sampler, batch size and model's tokenizer.  
    """
    input_ids, attention_masks = encode_documents_for_inference(documents, tokenizer, MAX_TOKENS_NUMBER)
    dataset = TensorDataset(input_ids, attention_masks)
    
    dataloader = DataLoader(
            dataset, 
            sampler = sampler(dataset), 
            batch_size = batch_size 
    )
    
    return dataloader


def encode_documents_for_inference(documents: np.ndarray, tokenizer: AutoTokenizer, MAX_TOKENS_NUMBER: int) -> tuple:
    """
    This method tokenises each document with the given model tokenizer. It uses padding 
    and truncation if the documents are too short or too long respectively. 
    Tokenised document representations are returned. 
    """
    input_ids = []
    attention_masks = []
    
    for document in documents:
        encoded_dict = tokenizer.encode_plus(
                            document,                      
                            add_special_tokens = True, 
                            max_length = MAX_TOKENS_NUMBER,           
                            padding='max_length',
                            return_attention_mask = True,  
                            truncation=True,
                            return_tensors = 'pt',    
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def batch_predict_for_inference(batch: torch.Tensor, device: torch.device, model: AutoModelForSequenceClassification) -> np.ndarray:
    """
    Generate and return model predictions for the batch of tweets.
    """
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        result = model(b_input_ids, 
                     attention_mask=b_input_mask,
                     return_dict=True)

    logits = result.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    logits = logits.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    
    return logits, probs


def make_predictions_for_inference(dataloader: DataLoader, device: torch.device, model: AutoModelForSequenceClassification) -> np.ndarray:
    """
    This method makes paper acceptance or rejection predictions. Dataloader
    is iterated through to predict the results of a batch of documents to not fill 
    the whole available memory.  
    """
    model.eval()

    predictions, probabilities = [], []

    for batch in dataloader:
        logits, probs = batch_predict_for_inference(batch, device, model)
        predictions.append(logits)
        probabilities.append(probs)
    
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    flat_probabilities = np.concatenate(probabilities, axis=0)

    return flat_predictions, flat_probabilities


def inference_for_single_document(document_json_as_dict: dict, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, MAX_TOKENS_NUMBER: int, 
                                  device: torch.device, batch_size: int=1) -> None:
    """
    This method makes predictions for the whole dataset of documents. Dataloader is created 
    and iterated through to predict the results of a batch of documents.
    """
    # "metadata" property is not generated on the parsed pdf files on the fly from the server.  
    if "metadata" in document_json_as_dict:
        document = parse_paper_from_json(document_json_as_dict, add_paper_metadata=True, mode=0, MAX_TOKENS_NUMBER=MAX_TOKENS_NUMBER)
    else:
        document = parse_paper_from_pdf_document(document_json_as_dict, add_paper_metadata=True, mode=0, MAX_TOKENS_NUMBER=MAX_TOKENS_NUMBER)
        
    # Create a dataloader from the loaded data
    dataloader = create_dataloader_for_inference(np.array([document]), SequentialSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)    

    predictions, probabilities = make_predictions_for_inference(dataloader, device, model)

    # Obtain the prediction and the probability for each decision
    document_probabilities, prediction = probabilities[0], predictions[0]

    return {'Denied': float(document_probabilities[0]), 'Accepted': float(document_probabilities[1])}
