# BERT Fine-Tuning for Sentence Classification (CoLA Dataset)

## Project Overview

This project fine-tunes a BERT-based model for sentence classification using the **CoLA (Corpus of Linguistic Acceptability)** dataset. The goal is to leverage the transformer-based architecture of BERT to classify sentences based on their linguistic acceptability. The model has been trained and evaluated using **PyTorch** and Hugging Face's `transformers` library.

## Features

- Utilizes `bert-base-uncased` for transfer learning.
- Implements tokenization using Hugging Face's `AutoTokenizer`.
- Fine-tunes the model using a **custom PyTorch training loop** instead of the `Trainer` API.
- Supports evaluation metrics such as **accuracy, F1-score, and Matthews Correlation Coefficient (MCC).**
- Handles dataset loading, preprocessing, and training pipeline seamlessly.

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Dikshitharyana/NLP-Sentence-classification-using-BERT.git 
cd NLP-Sentence-classification-using-BERT
```

### Install Dependencies

```bash
pip install torch transformers datasets scikit-learn
```

## Dataset

This project uses the **CoLA dataset**, which is part of the **GLUE benchmark**.

## Training the Model

The model is trained using the following parameters:

- **Epochs**: 4
- **Learning Rate**: 2e-5
- **Epsilon**: 1e-8
- **Optimizer**: AdamW

These parameters are set in the **Jupyter Notebook file** (`Sentence_classification_by_finetuning_BERT.ipynb`). Modify them as needed to experiment with different hyperparameters.

## Evaluation

To evaluate the model on the test dataset, run the evaluation cells in the notebook. This will generate metrics such as **Accuracy** and **MCC score**.

## Results

The model achieves the following performance on the CoLA test dataset:

- **Validation Accuracy**: 83%
- **Validation Loss**: 0.57
- **MCC Score**: 0.54

## Future Improvements

- Experiment with different BERT variants like **DistilBERT** or **RoBERTa**.
- Perform **hyperparameter tuning** using grid search or Bayesian optimization.
- Implement **data augmentation techniques** to improve generalization.

## License

This project is open-source and available under the **MIT License**.

## Acknowledgments

- **Hugging Face** for the `transformers` library
- **PyTorch** for deep learning framework
- **The GLUE Benchmark** for providing the CoLA dataset

## Contact

For queries, feel free to reach out via **email** or open an **issue on GitHub**!

