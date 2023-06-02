# multilayer_perceptron

## Project Description

This project implements a multilayer perceptron (MLP) classifier using the Scikit-learn library. The MLP is a type of feedforward artificial neural network that consists of multiple layers of nodes, including an input layer, one or more hidden layers, and an output layer. The MLP is trained using the backpropagation algorithm and can be used for classification tasks.

## Index

- Built With
- Prerequisites
- Getting Started
- Usage
- Algorithm
- Data
- Feedback
- Contribution
- Author
- License

## Built With

The project was built using the following libraries:

- Pandas
- Scikit-learn

## Prerequisites

Before running the project, ensure you have the following prerequisites:

- Python
- Pandas 
- Scikit-learn

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/VishalKumar-GitHub/multilayer_perceptron.git`
2. Navigate to the project directory: `cd multilayer_perceptron`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the project: `python main.py`

## Usage

To use the project, follow these guidelines:

1. Import the necessary libraries:

   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.neural_network import MLPClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   ```

2. Load the dataset:

   ```python
   df = pd.read_csv("mnist.csv")
   df = df.set_index('id')
   df.head()
   ```

3. Split the dataset into training and testing sets:

   ```python
   X = df.drop('class', axis=1)  # Assuming 'class' column contains the target variable
   y = df['class']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. Create and train the MLP classifier:

   ```python
   mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
   mlp.fit(X_train, y_train)
   ```

5. Make predictions on the test set and evaluate the performance:

   ```python
   y_pred = mlp.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred, average='macro')
   recall = recall_score(y_test, y_pred, average='macro')
   f1 = f1_score(y_test, y_pred, average='macro')

   print("Accuracy:", accuracy)
   print("Precision:", precision)
   print("Recall:", recall)
   print("F1-score:", f1)
   ```

## Algorithm

The project uses the multilayer perceptron (MLP) algorithm for classification tasks. The MLP is a feedforward artificial neural network with multiple layers of nodes. It is trained using the backpropagation algorithm.

## Data

The project uses the "mnist.csv" dataset for training and testing the MLP classifier. The dataset is loaded using the Pandas library and contains a set of features and corresponding class labels. The "class" column is assumed to contain the target variable.

## Feedback

Feedback is welcome! If you have any suggestions, issues, or feature requests, please [open an issue](https://github.com/VishalKumar-GitHub/multilayer_perceptron/issues) on the project's GitHub repository.

## Contribution

Contributions are welcome! If you would like to contribute to the project, please follow the guidelines for submitting pull requests.

## Author

- Author: Vishal Kumar
- GitHub: [VishalKumar-GitHub](https://github.com/VishalKumar-GitHub)

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
