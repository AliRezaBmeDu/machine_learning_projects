# Spam Detector Using TensorFlow in Python

## Overview
This repository contains the implementation of a spam detector using TensorFlow in Python. The project aims to classify emails as spam or ham (non-spam) using text classification techniques. It includes data preprocessing, model development using TensorFlow, and evaluation of the model's performance.

## Table of Contents
1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [Text Representation](#text-representation)
4. [Model Development](#model-development)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

## Dataset
The dataset(https://www.kaggle.com/datasets/venky73/spam-mails-dataset) used in this project contains emails labeled as spam or ham. It consists of 5171 samples with two columns: 'text' containing the email content and 'spam' indicating whether the email is spam (1) or ham (0). The dataset has been preprocessed to handle class imbalance by downsampling the majority class.

## Preprocessing
- Removed punctuations and stopwords from the text data.
- Converted text to lowercase to ensure consistency.

## Text Representation
- Tokenized the text data and converted words to sequences of token IDs.
- Padded sequences to ensure uniform length for model input.

## Model Development
- Implemented a Sequential model in TensorFlow consisting of:
  - Embedding Layers to learn vector representations of input tokens.
  - LSTM layer to capture patterns in sequence data.
  - Fully connected layers with ReLU activation.
  - Output layer with sigmoid activation for binary classification.
- Used binary cross-entropy loss and Adam optimizer for model training.

## Model Training and Evaluation
- Split the data into training and testing sets with an 80:20 ratio.
- Trained the model on the training data for 20 epochs with a batch size of 32.
- Evaluated the model on the test data, achieving an accuracy of approximately 97.83%.

## Results
- Plotted a graph to visualize the training and validation accuracies across epochs.

## Usage
To use this project, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Run the provided Python script to preprocess the data, train the model, and evaluate its performance.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.

## License
This project is licensed under the [MIT License](../LICENSE).


