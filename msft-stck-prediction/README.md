# Microsoft Stock Prediction

This project aims to predict the closing stock prices of Microsoft using LSTM (Long Short-Term Memory) neural networks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this project, we utilize LSTM neural networks to predict the closing stock prices of Microsoft. LSTM networks are a type of recurrent neural networks (RNN) that are capable of learning long-term dependencies, making them suitable for time-series prediction tasks.

## Dataset
The dataset used in this project is the Microsoft Stock [dataset](https://drive.google.com/file/d/1sDHtyrUTQoVr877zNBrGhSOZCsvV5BlY/view), which contains historical stock prices of Microsoft. The dataset includes features such as date, open price, close price, and volume.

## Project Structure
The project structure is organized as follows:
- `README.md`: This file provides an overview of the project.
- `main.py`: Python script for data preprocessing, model training, and prediction.
- `datasets/`: Directory containing the Microsoft Stock dataset (`MicrosoftStock.csv`).
- `results/`: Directory to store model evaluation results and visualizations.

## Installation
1. Clone the repository:
```sh
    https://github.com/AliRezaBmeDu/machine_learning_projects.git
```

2. Install the required dependencies:
- Tensorflow
- Pandas
- Seaborn
- Matplotlib

## Usage
1. Navigate to the project directory:
cd machine_learning_projects/microsoft-stock-prediction

2. Run the main script:
```sh
    python stock_prediction_using_ml.py
```


## Results
The model achieves promising results in predicting Microsoft's stock prices. Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to assess the model's performance.

## Contributing
Contributions to this project are welcome! To contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
