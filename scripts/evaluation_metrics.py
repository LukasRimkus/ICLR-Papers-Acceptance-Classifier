import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay


def plot_losses(training_stats: dict) -> None:
    """
    This method plots a graph showing how the losses of both the training and validation datasets
    changes over epochs. A separate line is drawn for each training and validation statistics.
    The data is taken from the training stats dictionary.
    """
    df_stats = pd.DataFrame(data=training_stats)

    train_dataset_loss, test_dataset_loss = df_stats['Training Loss'], df_stats['Valid. Loss']
    
    epochs_number = len(training_stats)
    
    # Plot the learning curve.
    plt.plot(np.arange(1, epochs_number+1), train_dataset_loss, 'b-o', label="Training")
    plt.plot(np.arange(1, epochs_number+1), test_dataset_loss, 'g-o', label="Validation")
    
    # Plot the labels with respective loss for each data point
    for i, (train_loss, test_loss) in enumerate(zip(train_dataset_loss, test_dataset_loss)):
        train_label = f"{train_loss:.3f}"

        plt.annotate(train_label, 
             (i+1, train_loss), 
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center') 
        
        test_label = f"{test_loss:.3f}"
        
        plt.annotate(test_label, 
             (i+1, test_loss),
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center') 

    
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(1, epochs_number+1))
    plt.grid()
    
    plt.show()


def plot_accuracies(training_stats: dict) -> None:
    """
    This method plots a graph showing how the accuracies of both the training and validation datasets
    changes over epochs. A separate line is drawn for each training and validation statistics.
    The data is taken from the training stats dictionary.
    """
    df_stats = pd.DataFrame(data=training_stats)

    train_dataset_accuracy, test_dataset_accuracy = df_stats['Training Accur.'], df_stats['Valid. Accur.']
    
    epochs_number = len(training_stats)
    
    # Plot the learning curve.
    plt.plot(np.arange(1, epochs_number+1), train_dataset_accuracy, 'b-o', label="Training")
    plt.plot(np.arange(1, epochs_number+1), test_dataset_accuracy, 'g-o', label="Validation")
    
    # Plot the labels with respective accuracy for each data point
    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_dataset_accuracy, test_dataset_accuracy)):
        train_label = f"{train_accuracy:.3f}"

        plt.annotate(train_label, 
             (i+1, train_accuracy), 
             textcoords="offset points",
             xytext=(0,10), # distance from text to points (x,y)
             ha='center') 
        
        test_label = f"{test_accuracy:.3f}"
        
        plt.annotate(test_label, 
             (i+1, test_accuracy),
             textcoords="offset points",
             xytext=(0,10),
             ha='center') 

    # Label the plot.
    plt.title("Training & Validation Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(range(1, epochs_number+1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid()
    
    plt.show()


def print_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
    """
    This method prints these model evaluation metrics: Accuracy, Precision, Recall and F1 Score. 
    """
    accuracy, precision, recall, f1_value = return_metrics(true_labels, predicted_labels) 
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_value:.4f}")


def return_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> tuple:
    """
    This method returns these model evaluation metrics: Accuracy, Precision, Recall and F1 Score. 
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1_value = f1_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1_value

    
def display_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
    """
    This method takes true and predicted labels and plots a confusion matrix on that.
    """
    display_labels = ["Rejected", "Accepted"]

    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def display_classification_report(true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
    """
    This method prints the classification report.
    """
    print(classification_report(y_true=true_labels, y_pred=predicted_labels, labels=[0, 1], digits=4))
    

def display_model_performance_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
    """
    This method prints various metrics for doing model final evaluation in terms of true and predicted labels.
    """
    print('Model Performance metrics:\n')
    print_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    print('\n\nModel Classification report:')
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels)

    print('\n\nPrediction Confusion Matrix:')
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels)


def plot_ensemble_accuracy_per_number_of_models(number_of_models: int, ensemble_performance_on_number_of_models: np.ndarray, models_mean_performance_on_number_of_models: np.ndarray, models_std_performance_on_number_of_models: np.ndarray) -> None:
    """
    This method plots a graph showing how the accuracies of mean individual models (depending on the x axis 
    which denotes the number of models so far) and ensemble performance change over the trained number
    of models. A separate line is drawn for each.
    """
    x_axis_number_of_models = np.arange(1, number_of_models + 1, 1)
    
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 13})
    
    plt.plot(x_axis_number_of_models, ensemble_performance_on_number_of_models, 'b-o', label="Ensemble Accuracy")
    plt.plot(x_axis_number_of_models, models_mean_performance_on_number_of_models, 'g-o', label="Mean Model Accuracy")
    
    # variable to decide if the label of a point on a graph is up or down in regards to a plotted point
    up_or_down = 1

    # Plot the labels with respective accuracy for each data point
    for i, (ensemble_accuracy, mean_models_accuracy) in enumerate(zip(ensemble_performance_on_number_of_models, models_mean_performance_on_number_of_models)):
        
        if i % 2 == 0:
            train_label = f"{ensemble_accuracy:.3f}"
            
            plt.annotate(train_label,
                 (i+1, ensemble_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center') 

        else:
            test_label = f"{mean_models_accuracy:.3f}"

            plt.annotate(test_label, 
                 (i+1, mean_models_accuracy),
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center')
        
        if i == number_of_models - 1:
            train_label = f"{ensemble_accuracy:.3f}"
            
            plt.annotate(train_label,
                 (i+1, ensemble_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center') 
            
            test_label = f"{mean_models_accuracy:.3f}"

            plt.annotate(test_label, 
                 (i+1, mean_models_accuracy),
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center')

        
        up_or_down = up_or_down * (-1) if up_or_down < 0 else up_or_down // (-1)

    plt.xlabel("Number of Models")
    plt.ylabel("Accuracy")

    plt.xticks(range(1, number_of_models+1))
    
    plt.legend()
    plt.grid()

    plt.show()
    

def show_ensemble_model_accuracies(accuracies: np.ndarray) -> None:
    """
    This method goes through model achieved accuracies and displays them.
    """
    for i, accuracy in enumerate(accuracies):
        print(f"Model {i+1} accuracy: {accuracy*100:.2f}%")

    print(f"\nMean models accuracy: {accuracies.mean()*100:.2f}% +- {accuracies.std()*100:.2f}%")
