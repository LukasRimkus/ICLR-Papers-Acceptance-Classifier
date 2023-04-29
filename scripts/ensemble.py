import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import RandomSampler, SequentialSampler

import random
import os

import numpy as np
import pandas as pd

from scripts.data_preprocessing import create_dataloader
from scripts.evaluation_metrics import display_model_performance_metrics, plot_accuracies, plot_losses
from scripts.utils import save_model
from scripts.test_models import make_predictions
from scripts.train_models import train


NUMBER_OF_CLASSES = 2


def train_ensemble_models(number_of_models: int, model_name: str, tokenizer: AutoTokenizer, training_documents: np.ndarray,
                          training_labels: np.ndarray, val_documents: np.ndarray, val_labels: np.ndarray, batch_size: int,
                          learning_rate: float, weight_decay: float, epochs: int, device: torch.device, MAX_TOKENS_NUMBER: int,
                          ensemble_path: str, random_state: int) -> tuple:
    """
    This method defines the pipeline of training the ensemble. Given number of models 
    is trained, their results are preserved and returned. As well, all models performance 
    are combined to an ensemble using the probability foe each paper averaging to hopefully 
    increase the performance compared to a single model. 
    Each model is trained after taking a random (with a set seed) sample with replacement (bootstrap) 
    from a training data set.
    """
    # Directory of stored ensemble models
    ensemble_folder_name = f"{model_name.replace('/', '-')}_lr_{learning_rate}_batch_{batch_size}_weightdecay_{weight_decay}"

    # Create dataloaders for datasets
    validation_dataloader = create_dataloader(val_documents, val_labels, SequentialSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)

    # Init arrays for storing statistics for each model
    model_eval_probabilities = np.zeros([number_of_models, len(val_documents), NUMBER_OF_CLASSES])
    model_eval_accuracies = np.zeros([number_of_models])
    
    # Store statistics of the ensemble performance continuously when training new models
    ensemble_performance_on_number_of_models = np.zeros([number_of_models])
    models_mean_performance_on_number_of_models = np.zeros([number_of_models])
    models_std_performance_on_number_of_models = np.zeros([number_of_models])
    
    # Init seeds for each model
    seeds = np.arange(random_state, random_state + number_of_models, 1)

    # Check if the folder for all Ensemble models exists
    if not os.path.exists(ensemble_path):
        os.mkdir(ensemble_path)

    ensemble_folder_path = os.path.join(ensemble_path, ensemble_folder_name)

    # Check if the folder for this Ensemble model already exists
    if not os.path.exists(ensemble_folder_path):
        os.mkdir(ensemble_folder_path)

    for i in range(number_of_models):
        print(f"================ Ensemble model #{i+1} ================")
        seed_val = seeds[i]
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
        # Define a model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = NUMBER_OF_CLASSES, 
            output_attentions = False, 
            output_hidden_states = False
        )

        model = model.to(device)

        # Do a bootstrap with replacement
        data_df = pd.DataFrame(data={"Docs": training_documents, "Decision": training_labels})
        rejected_papers_df = data_df[data_df.Decision == 0].copy()
        accepted_papers_df = data_df[data_df.Decision == 1].copy()

        rejected_papers_df = rejected_papers_df.sample(frac=1, random_state=seed_val, replace=True)
        accepted_papers_df = accepted_papers_df.sample(frac=1, random_state=seed_val, replace=True)

        sampled_data_df = pd.concat([accepted_papers_df, rejected_papers_df])
        
        # Take bootstrapped data for each model
        documents_train = sampled_data_df.Docs.values
        labels_train = sampled_data_df.Decision.values

        train_dataloader = create_dataloader(documents_train, labels_train, RandomSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)
        
        total_steps = len(train_dataloader) * epochs

        # Create an AdamW optimizer and linear scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        
        # Train and evaluate with validation set the model
        model, training_stats_for_model = train(total_steps, epochs, model, train_dataloader, device, optimizer, scheduler, validation_dataloader)
        
        # Extract model predictions and metrics
        epochs_number = len(training_stats_for_model)
        predicted_labels = training_stats_for_model[epochs_number-1]['Evaluation Predictions']
        probabilities = np.array(training_stats_for_model[epochs_number-1]["Probabilities"])
        model_accuracy = training_stats_for_model[epochs_number-1]["Valid. Accur."]

        model_eval_probabilities[i] = probabilities
        model_eval_accuracies[i] = model_accuracy

        # Plot the model performance in terms of loss and accuracy 
        plot_losses(training_stats_for_model)
        plot_accuracies(training_stats_for_model)

        # Save the model in local storage (those checks are essential for Google Colab)
        output_path = os.path.join(ensemble_folder_path, f"model_{i+1}")
        save_model(output_path, model, tokenizer)

        # Obtain the accuracy reached by the model so far
        ensemble_eval_accuracy, ensemble_predictions = get_ensemble_accuracy(model_eval_probabilities[:i+1], val_documents, val_labels)

        ensemble_performance_on_number_of_models[i] = ensemble_eval_accuracy

        # Calculate ensemble performance so far after taking (i + 1) models
        models_mean_performance_on_number_of_models[i] = model_eval_accuracies[:i+1].mean()
        models_std_performance_on_number_of_models[i] = model_eval_accuracies[:i+1].std()
    
    print(f"Display Ensemble validation performance metrics after training {number_of_models} models")
    display_model_performance_metrics(val_labels, ensemble_predictions)
    
    return model_eval_accuracies, ensemble_performance_on_number_of_models, models_mean_performance_on_number_of_models, models_std_performance_on_number_of_models
    
    
def get_ensemble_accuracy(model_probabilities: np.ndarray, documents: np.ndarray, labels: np.ndarray) -> tuple:
    """
    This method computes the ensemble performance using majority voting (averaging probabilities 
    for each document) given current number of models. 
    """
    # To make it easier to add up probabilities for each document
    # instead of having an array of model document predictions, I have an array of document predictions for models
    # which can be more easily used to iterate through predictions
    model_probabilities_transposed = np.transpose(model_probabilities, (1, 0, 2))
    final_predictions = np.zeros([len(documents)])

    for j, model_predictions in enumerate(model_probabilities_transposed):
        probabilities = np.zeros(3)
        
        # For each document, go through each model's prediction in probabilities, which are summed over here.
        for probs in model_predictions:
            for label, class_prob in enumerate(probs):
                probabilities[label] += class_prob
        
        # One which shows highest probability aggregation is chosen by an ensemble
        final_predictions[j] = np.argmax(probabilities)
    
    ensemble_accuracy = (final_predictions == labels).sum()/len(labels)
    return ensemble_accuracy, final_predictions


def test_ensemble_models(number_of_models: int, tokenizer: AutoTokenizer, model_name: str, testing_documents: np.ndarray, 
                         testing_labels: np.ndarray, learning_rate: float, weight_decay: float, batch_size: int, device: 
                         torch.device, MAX_TOKENS_NUMBER: int, ensemble_path: str) -> tuple:
    """
    This method computes the metrics for all trained models and produce the final 
    performance of an ensemble which can be compared to the mean of models performance
    without combining their "votes". 
    """
    # Directory of stored ensemble models
    ensemble_folder_name = f"{model_name.replace('/', '-')}_lr_{learning_rate}_batch_{batch_size}_weightdecay_{weight_decay}"

    # Init arrays to contain the metrics achieved so far 
    model_test_probabilities = np.zeros([number_of_models, len(testing_documents), NUMBER_OF_CLASSES])
    model_test_accuracies = np.zeros([number_of_models])
    
    ensemble_performance_on_number_of_models = np.zeros([number_of_models])
    models_mean_performance_on_number_of_models = np.zeros([number_of_models])
    models_std_performance_on_number_of_models = np.zeros([number_of_models])

    # Create a dataloader of testing data
    testing_dataloader = create_dataloader(testing_documents, testing_labels, SequentialSampler, batch_size, tokenizer, MAX_TOKENS_NUMBER)    

    for i in range(number_of_models):
        # Load a stored model
        output_path = f"{ensemble_path}/{ensemble_folder_name}/model_{i+1}"
        model = AutoModelForSequenceClassification.from_pretrained(output_path)
        model = model.to(device)

        # Obtain the predictions metrics for the trained model  
        accuracy, _, _, _, probs = make_predictions(testing_dataloader, device, model)

        model_test_probabilities[i] = probs
        model_test_accuracies[i] = accuracy

        # Find the ensemble accuracy so far with current number of models
        ensemble_test_accuracy, ensemble_predictions = get_ensemble_accuracy(model_test_probabilities[:i+1], testing_documents, testing_labels)

        ensemble_performance_on_number_of_models[i] = ensemble_test_accuracy
        print(f"\nEnsemble performance: {ensemble_test_accuracy*100:.2f}%")

        models_mean_performance_on_number_of_models[i] = model_test_accuracies[:i+1].mean()
        models_std_performance_on_number_of_models[i] = model_test_accuracies[:i+1].std()
    
    print(f"\nDisplay Ensemble test performance metrics after training {number_of_models} models")
    display_model_performance_metrics(testing_labels, ensemble_predictions)

    return model_test_accuracies, ensemble_performance_on_number_of_models, models_mean_performance_on_number_of_models, models_std_performance_on_number_of_models
