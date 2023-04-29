# Reviewer 2 Must Be Stopped!

This project involved constructing a binary sequence classifier to determine whether a scientific paper submitted to a conference was rejected or accepted. As described in the proposal, we utilised state-of-the-art transformer models for this task: BERT, RoBERTa and Longformer. However, they have a limitation of having a limit of how long the inputs can be (however, Longformer has a limit of 4096, whereas BERT and RoBERTa only 512 tokens), which was mitigated by truncating the papers in these ways:
1. `INTRODUCTION_WITH_ABSTRACT` - start from the abstract till the tokens limit is reached.
2. `INTRODUCTION_WITHOUT_ABSTRACT` - skip the abstract and start from the introduction section.
3. `MIDDLE` - take the middle of the paper.
4. `TAIL` - take the span of text from the end of the paper, including the conclusion.
5. `ABSTRACT_WITH_TAIL` - take the parts of the abstract and tail (including the conclusion).

In addition to what we initially proposed, we built a Bagging Ensemble, which combines the models of the same architecture and hyperparameters (only the training dataset is bootstrapped for each model) to test if it can improve the performance of a single trained model.

The entire project is designed to be run from notebooks. However, they only contain high-level code, as the concrete implementations can be found in the `scripts` folder. More details and instructions on how to run notebooks are provided in the notebooks.

-------------------------
# Notebooks

There are four notebooks in total:  
- `ModelTrainingAndTesting.ipynb` provides code for training and evaluating a single model's performance.
- `HyperparameterOptimisation.ipynb` sets up the Weights & Biases Hyperparameter Selection Sweep over selected parameters and ranges of values.
- `EnsembleTraining.ipynb` contains the code to train and evaluate a selected number of models which make up a Bagging Ensemble.
- `Demo.ipynb` can be used for making inferences on any selected paper if it is already parsed to an acceptable JSON structure as in the dataset or if it is a pdf file which can be parsed using our parser hosted on Cloud.

-------------------------
# Dataset

We utilized the ASAP-Review dataset for this project. The dataset can be found [here](https://drive.google.com/file/d/1nJdljy468roUcKLbVwWUhMs7teirah75/view). More about the dataset structure and potential use cases are discussed [here](https://jair.org/index.php/jair/article/view/12862).

Comments on data selection and pre-processing:
- We focused on papers submitted to ICLR between 2017 and 2020.
- NeurIPS papers were not used, as only accepted papers are included from NeurIPS.
- The `aspect_data` folder, containing paper peer review annotations, was excluded as it is irrelevant to this project.
- There are two essential JSON files for each paper, namely `{PAPER}_content.json` (contains the parsed paper contents) and `{PAPER}_paper.json` (contains the paper metadata like the acceptance or rejection decision). They are used for reading the data for model training and evaluation. 
- The papers from ICLR contain about twice more rejected than accepted ones. Therefore the dataset was subsampled to have an equal data split for each class.
- The data was split into train/val/test in the code 20% of papers allocated for testing, whereas the remaining documents were divided into another 80% and 20% for the train and validation sets respectively.

-------------------------
# Model Architectures

The following pre-trained models from Hugging Face were fine-tuned in this project:

1. [BERT Base (cased)](https://huggingface.co/bert-base-cased)
2. [RoBERTa Base](https://huggingface.co/roberta-base)
3. [Longformer Base 4096](https://huggingface.co/allenai/longformer-base-4096)

The trained best models can be found [here](https://drive.google.com/drive/folders/1Lb1NpoTninBxQUCldKMInz_RI4FeHdxP?usp=share_link). 
