# REPORT

## 1. Project Title
IMDB Sentiment Classification using Classical and Deep Learning Methods

## 2. Problem Definition
In this project, we focused on binary sentiment classification using the IMDB movie review dataset. The goal was to classify movie reviews as either positive or negative. Rather than building a broad project, we aimed to apply and compare the methodologies covered in the course on a focused problem.

## 3. Dataset Description
We used the IMDB movie review dataset. The dataset contains text reviews and sentiment labels. The labels are binary:
- 0 = negative
- 1 = positive

The dataset was suitable for this project because it directly supports binary classification and allows us to compare both classical machine learning methods and neural network-based approaches on the same task.

## 4. Data Preparation
The dataset was first inspected to understand its structure and class balance. The main text column was used as input, and the sentiment column was used as the target.

We split the labeled data into:
- training set
- validation set
- test set

This allowed us to train the models, monitor their performance during development, and evaluate final performance on unseen data.

Since machine learning models cannot directly process raw text, we transformed the review texts into numerical representations using TF-IDF. We set the maximum number of features to 10,000 and used English stop word removal.

## 5. Methodologies Applied

### 5.1 Logistic Regression
We first used Logistic Regression as a baseline model. This gave us a simple and strong reference point for the sentiment classification task. It also helped us compare whether more complex neural network models would actually improve performance.

### 5.2 Basic MLP
We then implemented a basic Multi-Layer Perceptron (MLP) model. This model used the TF-IDF features as input and applied a feedforward neural network structure with ReLU activation. Since the problem is binary classification, we used BCEWithLogitsLoss.

### 5.3 MLP with Dropout
After observing that the basic MLP showed signs of overfitting, we added dropout regularization. The aim was to reduce memorization and improve generalization performance.

### 5.4 MLP with L2 Regularization
We also tested L2 regularization by applying weight decay during optimization. This was used as another regularization strategy to control model complexity.

### 5.5 MLP with Early Stopping
We applied early stopping to stop training when validation performance no longer improved. This was done to prevent unnecessary training and reduce overfitting. This approach gave the best final performance among all tested models.

### 5.6 Optimizer Comparison (Adam vs SGD)
To include the optimization methodologies covered in the course, we compared Adam and SGD on the same MLP-based setup. This allowed us to observe how optimizer choice affects learning speed and final model quality.

### 5.7 MLP with BatchNorm and He Initialization
To further address training stability, we implemented a model that included Batch Normalization and He initialization. This was done to represent additional deep learning optimization and stability concepts covered in the lectures.

## 6. Hyperparameter Choices
The main hyperparameters we used were selected to keep the experiments simple, comparable, and suitable for the course scope.

Examples of hyperparameter choices:
- TF-IDF max features: 10,000
- Batch size: 64
- Hidden layer size: 128
- Learning rate: 0.001 for Adam
- Number of epochs: 5 or 15 depending on the experiment
- Dropout rate: 0.5
- L2 regularization: weight decay = 1e-4
- Early stopping patience: 2

These values were not chosen randomly. They were selected as reasonable starting points for a text classification problem and then evaluated through validation performance.

## 7. Performance Comparison

| Model | Test Accuracy |
|------|---------------:|
| Logistic Regression | 0.8828 |
| Basic MLP | 0.8684 |
| MLP + Dropout | 0.8808 |
| MLP + L2 | 0.8672 |
| MLP + Dropout + Early Stopping | **0.8936** |
| Adam MLP | 0.8796 |
| SGD MLP | 0.7356 |
| MLP + BatchNorm + He + Dropout + Early Stopping | 0.8832 |

## 8. Best Model Selection and Rationale
The best-performing model was **MLP + Dropout + Early Stopping**, with a test accuracy of **0.8936**.

Our observations were as follows:
- Logistic Regression was a strong baseline and performed better than some neural network variants.
- The basic MLP showed overfitting behavior.
- Dropout improved the MLP and brought it close to the baseline.
- L2 regularization did not improve performance in our setup.
- Early stopping gave the best result by preventing the model from training past its best validation point.
- Adam clearly outperformed SGD in our experiments.
- BatchNorm and He initialization were tested for training stability, but they did not outperform the best early-stopped dropout model.

For these reasons, we selected **MLP + Dropout + Early Stopping** as the final model.

## 9. Conclusion
This project was designed to focus on methodological depth rather than breadth. Using the IMDB sentiment classification task, we applied several concepts covered in the course, including train/validation/test splitting, TF-IDF feature extraction, baseline modeling, neural networks, regularization methods, optimizer comparison, and training stability techniques.

The results showed that more complex models do not always outperform simpler baselines unless regularization and training control are properly applied. Among all methods, MLP with Dropout and Early Stopping gave the best overall performance.

This project allowed us to compare classical and deep learning approaches on the same dataset and make a justified final model selection based on experimental results.