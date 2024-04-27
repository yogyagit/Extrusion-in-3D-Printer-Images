# Extrusion-in-3D-Printer-Images
Contains the code and the trained model for this task using a Vision Transformer
https://drive.google.com/drive/folders/1AlkKkih9dm3HKUhCOkCRjjyKlQCSmYez?usp=share_link

Drive link has the Model Weights.


# Custom Vision Transformer for Image Classification

This repository contains a PyTorch implementation of a custom Vision Transformer (ViT) model tailored for classifying images based on specific features such as under-extrusion in 3D printing. It includes scripts for dataset preparation, model training, validation, testing, and prediction output preparation.

## Files and Directories

- `train_val_test.py`: Main script for training, validating, and testing the model.
- `custom_dataset.py`: Contains custom dataset classes for training and testing.
- `/scratch/ag8766/yogya/`: Directory containing the datasets and image files (not included in this repository).

## Setup

### Dependencies

- Python 3.8+
- PyTorch 1.7+
- torchvision
- PIL
- tqdm
- pandas
- scikit-learn

### Installation

Install the required libraries using pip:

```bash
pip install torch torchvision pandas tqdm pillow scikit-learn
```

## Usage

### Data Preparation

Ensure your data is in a CSV format with columns for img_path and has_under_extrusion, along with any other necessary identifiers which should be removed before training.

### Training the Model

Run the train_val_test.py script to start training and validation:

```bash
python train_val_test.py
```

This script will perform the following actions:

Load and preprocess the data.
Split the data into training, validation, and testing sets.
Define and train the custom Vision Transformer model.
Validate and save the best model based on validation accuracy.
Test the model with the test dataset and save predictions.

### Custom Dataset
The CustomDataset and CustomDatasetTest classes in custom_dataset.py are used to handle loading and transforming images for training and testing purposes. These classes are utilized by the DataLoader during the training process.

### Features
Custom Vision Transformer (ViT) Model: Integrates a pretrained ViT model with custom fully connected layers for specialized tasks.
Data Augmentation: Applies transformations such as random rotations and crops to enhance model generalizability.
Performance Metrics: Evaluates model performance using accuracy and F1-score.

## Output
The test predictions are saved in a CSV file named test_preds.csv, and a final submission file submission_7.csv is prepared for further analysis or competition submission.

### Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

