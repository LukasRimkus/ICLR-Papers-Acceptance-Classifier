import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler
from transformers import AutoTokenizer


def encode_documents(documents: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer, MAX_TOKENS_NUMBER: int) -> tuple:
    """
    This method tokenises each document with the given model tokenizer. It uses padding 
    and truncation if the documents are too short or too long respectively. 
    Tokenised representations are returned. 
    """

    input_ids = []
    attention_masks = []

    for document in documents:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            document,                     
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_TOKENS_NUMBER, # Pad & truncate all sentences.
                            padding='max_length',
                            return_attention_mask = True,  
                            truncation=True,
                            return_tensors = 'pt', # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def create_dataloader(documents: np.ndarray, labels: np.ndarray, sampler: Sampler, batch_size: int, tokenizer: AutoTokenizer, MAX_TOKENS_NUMBER: int) -> DataLoader:
    """
    This method created a dataloader object given parameters of labels, max number of tokens, 
    type of the sampler, batch size and model's tokenizer.  
    """
    input_ids, attention_masks, labels = encode_documents(documents, labels, tokenizer, MAX_TOKENS_NUMBER)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    dataloader = DataLoader(
            dataset, 
            sampler = sampler(dataset), 
            batch_size = batch_size 
    )
    
    return dataloader