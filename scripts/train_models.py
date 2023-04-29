import numpy as np

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm.auto import tqdm

import time
import random

from scripts.evaluation_metrics import display_model_performance_metrics, plot_accuracies, plot_losses
from scripts.data_preprocessing import create_dataloader
from scripts.utils import format_time


NUMBER_OF_CLASSES = 2


# Training and evaluation code inspired by https://mccormickml.com/2019/07/22/BERT-fine-tuning/.
def train(total_steps: int, epochs: int, model: AutoModelForSequenceClassification, train_dataloader: DataLoader, device: torch.device, 
          optimizer: AdamW, scheduler: transformers.SchedulerType, validation_dataloader: DataLoader) -> tuple:
    """
    This method does the whole training and evaluation of the model given number of epochs.

    Training:
    - Unpack data inputs and class labels.
    - Load data onto the GPU for better acceleration compared to CPU.
    - Clear out the gradients calculated in the last pass. 
        - In pytorch the gradients accumulate by default (e.g. useful for RNNs) unless you explicitly clear them out which is needed here.
    - Forward pass (feed input data through the neural network).
    - Backpropagation pass.
    - Update parameters with optimizer.step().
    - Track variables for performance monitoring progress.
    """
    progress_bar = tqdm(range(total_steps))
    
    # Store a number of quantities such as training and validation loss, validation accuracy, and timings
    training_stats = []

    # Measure the total training time for the whole run
    start = time.time()

    for epoch_i in range(epochs):
        model, epoch_stats = train_epoch(model, train_dataloader, device, epoch_i, epochs, optimizer, scheduler, progress_bar, validation_dataloader)
        training_stats.append(epoch_stats)
        
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-start)))

    return model, training_stats


def train_epoch(model: AutoModelForSequenceClassification, train_dataloader: DataLoader, device: torch.device, epoch_i: int, epochs: int, 
                optimizer: AdamW, scheduler: transformers.SchedulerType, progress_bar: tqdm, validation_dataloader: DataLoader) -> tuple:
    """
    This method goes through one epoch of the model training. Training is done and evaluation is done 
    to calculate the training and evaluation loss and accuracy. Statistics of this epoch training 
    is returned. 
    """
    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')

    # Measure how long the training epoch takes
    t0 = time.time()

    # Reset the total loss for this epoch
    total_train_loss = 0
    total_train_accuracy = 0
    total_correct_predictions = 0
    total_labels = 0
    
    # Put the model into training mode. The call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    for step, batch in enumerate(train_dataloader):
        # Those parameters are changed by passing them by parameters and returning them to update the values 
        model, total_train_loss, total_train_accuracy, total_correct_predictions, total_labels = train_batch(batch, model, device, optimizer, 
                                                                                                             scheduler, progress_bar, total_train_loss, 
                                                                                                             total_train_accuracy, total_correct_predictions, total_labels)
                                                                                                             
                                                                                                             
    avg_train_loss = total_train_loss / len(train_dataloader)  # Calculate the average loss over all of the batches
    
    train_accuracy = float(total_correct_predictions / total_labels)

    # Measure how long this epoch took
    training_time = format_time(time.time() - t0)
    print("")
    print(f"\n  Average training loss: {avg_train_loss:.2f}")
    print(f"  Training epoch took: {training_time:}")
    print(f"  Training Accuracy: {train_accuracy:.2f}")

    # After the completion of each training epoch, measure the performance on the validation set
    print("\nRunning Validation...")
    last_epoch = (epoch_i + 1) == epochs
    total_eval_loss, t0, validation_accuracy, predicted_labels, probabilities = evaluate_results(validation_dataloader, model, device, last_epoch)

    # Report the final accuracy for this validation run
    print(f"  Validation Accuracy: {validation_accuracy:.2f}")

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took
    validation_time = format_time(time.time() - t0)

    print(f"Validation Loss: {avg_val_loss:.2f}")
    print(f"Validation took: {validation_time:}")

    # Record all statistics from this epoch
    epoch_stats = {
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        "Training Accur.": train_accuracy,
        'Valid. Loss': avg_val_loss,
        'Valid. Accur.': validation_accuracy,
        'Training Time': training_time,
        'Validation Time': validation_time,
        'Evaluation Predictions': predicted_labels,
        'Probabilities': probabilities
    }
    
    return model, epoch_stats


def train_batch(batch: torch.Tensor, model: AutoModelForSequenceClassification, device, optimizer, scheduler: transformers.SchedulerType, 
                progress_bar: tqdm, total_train_loss: float, total_train_accuracy: float, total_correct_predictions: int, total_labels: int) -> tuple:
    """
    This method trains one batch of documents. Model generates an output, loss is calculated, 
    a backwards pass is done, gradient is clipped, optimizer and scheduler are updated, gradient is cleaned
    and finally training accuracy is found.  
    """
    # Unpack this training batch from our dataloader. 
    # As we unpack the batch, we'll also copy each tensor to the device using the 
    # `to` method. Batch contains three pytorch tensors:

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    # Perform a forward pass (evaluate the model on this training batch).
    # In PyTorch, calling `model` will in turn call the model's `forward` 
    # function and pass down the arguments. The `forward` function is 
    # documented here: 
    # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
    # The results are returned in a results object, documented here:
    # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
    # Specifically, get the loss (because we provided labels) and the
    # "logits" -- the model outputs prior to activation.

    output = model(b_input_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels,
                    return_dict=True) 
    
    loss = output.loss
    logits = output.logits

    # Accumulate the training loss over all of the batches so that we can
    # calculate the average loss at the end. `loss` is a Tensor containing a
    # single value; the `.item()` function just returns the Python value 
    # from the tensor.
    total_train_loss += loss.item()

    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    # The optimizer dictates the "update rule"--how the parameters are
    # modified based on their gradients, the learning rate, etc.
    optimizer.step()

    # Update the learning rate.
    scheduler.step()

    # Clear any previously calculated gradients before performing a backward pass. PyTorch doesn't do this automatically because
    # accumulating the gradients is "convenient while training RNNs". 
    # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
    optimizer.zero_grad()

    # Calculate the training accuracy
    # This finds the the indices of the max values which correspond to classes [0, 1, 2]
    _, preds = torch.max(logits, dim=1)
    total_correct_predictions += torch.sum(preds == b_labels)
    total_labels += len(b_labels)

    progress_bar.update(1)
    
    return model, total_train_loss, total_train_accuracy, total_correct_predictions, total_labels


def evaluate_results(dataloader: DataLoader, model: AutoModelForSequenceClassification, device: torch.device, last_epoch: bool) -> tuple:
    """
    This method takes a model after training some number of epochs and uses it to evaluate
    its performance on the validation set. 

    Evalution:
    - Unpack data inputs and class labels.
    - Load data onto the GPU for better acceleration compared to CPU.
    - Forward pass (feed input data through the neural network).
    - Compute loss and accuracy on the validation data and track variables for performance monitoring progress.
    """
    t0 = time.time()

    # Put the model in evaluation mode because the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_loss = 0
    total_correct_predictions = 0
    total_labels = 0

    true_labels, predicted_labels, probabilities = list(), list(), list()

    # Evaluate data for one epoch
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backpropogation (training).
        with torch.no_grad():     
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            output = model(b_input_ids, 
                           attention_mask=b_input_mask,
                           labels=b_labels)

        # Get the loss and "logits" output by the model. The "logits" are the output values prior to applying an 
        # activation function like the softmax.
        loss = output.loss
        logits = output.logits
        
        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Move logits, probs and labels to CPU if we are using GPU
        logits = logits.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
        # find how many have been predicted correctly
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        total_correct_predictions += np.sum(pred_flat == labels_flat)
        total_labels += len(labels_flat)
        
        true_labels.extend(labels_flat)
        predicted_labels.extend(pred_flat)
        probabilities.extend(probs)
    
    # Display statistics of the whole epoch using graphs after reaching the last epoch
    if last_epoch:
        display_model_performance_metrics(true_labels, predicted_labels)
    
    eval_accuracy = total_correct_predictions / total_labels
    return total_eval_loss, t0, eval_accuracy, predicted_labels, probabilities


def train_transformer_model(model_name: str, epochs: int, batch_size: int, learning_rate: float, 
                            weight_decay: float, training_documents: np.ndarray, val_documents: np.ndarray,
                            training_labels: np.ndarray, val_labels: np.ndarray, device: torch.device, 
                            MAX_TOKENS_NUMBER: int, random_state: int) -> tuple:
                            
    """
    This method does training for the given different model
    (BERT, RoBERTa and Longformer) which can be compared with each other as they use the same
    hyperparameters.
    """
    print(f"============ Transformer {model_name} is now being tested ============\n")
    
    # Set the random seed
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels = NUMBER_OF_CLASSES, 
        output_attentions = False, 
        output_hidden_states = False
    )

    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataloaders for datasets
    train_dataloader = create_dataloader(training_documents, training_labels, RandomSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)
    validation_dataloader = create_dataloader(val_documents, val_labels, SequentialSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)

    total_steps = len(train_dataloader) * epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # Train the model and collect the training data
    model, training_stats_for_model = train(total_steps, epochs, model, train_dataloader, device, optimizer, scheduler, validation_dataloader)

    # Plot the training data about the loss and accuracy
    plot_losses(training_stats_for_model)
    plot_accuracies(training_stats_for_model)
    
    torch.cuda.empty_cache() 
    
    print(f"Training Accuracy: {training_stats_for_model[-1]['Training Accur.']:.4f}")
    print(f"Training Loss: {training_stats_for_model[-1]['Training Loss']:.4f}")
    print(f"Validation Accuracy: {training_stats_for_model[-1]['Valid. Accur.']:.4f}")
    print(f"Validation Loss: {training_stats_for_model[-1]['Valid. Loss']:.4f}")

    print(f"\n============ Transformer {model_name} has been successfully tested ============\n\n")
    
    return model, training_stats_for_model, tokenizer
    