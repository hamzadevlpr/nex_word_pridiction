# Next Word Prediction with LSTM using TensorFlow

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)

Predict the next word in a sentence using a Long Short-Term Memory (LSTM) neural network implemented with TensorFlow. This project serves as a simple example of natural language processing (NLP) and sequence prediction.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
In this project, we use an LSTM-based neural network to predict the next word in a sentence. The model is trained on a dataset of text sequences, and it learns to capture the patterns and relationships between words.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/hamzadevlpr/next-word-prediction.git
   cd next-word-prediction
Install the required dependencies:
 ```bash
  pip install -r requirements.txt
  ```

Usage
Preprocess your own dataset or use the provided example dataset.
Train the LSTM model using the provided script:
 ```bash
python train.py --data_path /path/to/your/dataset.txt
```

Use the trained model to predict the next word:

 ```bash
python predict.py --model_path /path/to/saved/model.h5 --input_text "your input text here"
```

Dataset
The dataset used for training should be a plain text file containing sequences of sentences. Each line in the file represents a sentence.

Example Dataset:

Model Architecture
The LSTM-based model architecture is defined in the model.py file. Feel free to modify the architecture to suit your specific needs.

Training
The training script (train.py) is used to train the LSTM model on your dataset. You can customize hyperparameters such as batch size, epochs, and learning rate within the script.

Results
After training, the model can be used to predict the next word in a given input text. Check the predict.py script for an example of how to use the trained model.

Contributing
Contributions are welcome! Fork the repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License.
