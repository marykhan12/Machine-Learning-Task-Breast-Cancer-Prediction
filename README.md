# Breast Cancer Prediction Using Machine Learning Algorithms
This project is to identify the effective and predictive algorithm for the detection of Breast cancer, therefore we applied machine learning classifiers Support Vector Machine (SVM), Random Forests, Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN) on Breast Cancer Wisconsin Diagnostic dataset and evaluate the results obtained to define which model provides a higher accuracy and also developed using Flask to provide easy interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Algorithms Used](#algorithms-used)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [Feedback](#feedback)

## Project Overview
Breast cancer has now overtaken lung cancer as the most commonly diagnosed cancer in women worldwide. Today, one in 5 people worldwide will develop cancer during their lifetime. Projections suggest that the number of people being diagnosed with cancer will increase still further in the coming years, and will be nearly 50% higher in 2040 than in 2020. The number of cancer deaths has also increased, from 6.2 million in 2000 to 10 million in 2020. More than one in six deaths is due to cancer.The successful introduction of Artificial Intelligence(AI) in medical practice is an important stake in the renovation of the health system and more precisely in cancer care. In this project, machine learning algorithms are applied to the well-known Breast Cancer Wisconsin dataset to predict whether a tumor is benign or malignant based on features extracted from a patient's medical data.

## Algorithms Used
The following machine learning algorithms are implemented and compared for breast cancer prediction:

1. **Support Vector Classifier (SVC)**: It is basically used for classification tasks and it seeks to find the hyperplane that best separates the data points in separate classes. 
2. **K-Nearest Neighbors (KNN)**: It is used for Classification as well as regression problem. It is a simple, instance-based learning algorithm that classifies a new data point based on the majority class among its nearest neighbors.
3. **Random Forest**: An ensemble method that constructs multiple decision trees and merges their results to improve accuracy and control overfitting.
4. **Logistic Regression**: it is a supervised machine learning algorithm that accomplishes binary classification tasks by predicting the probability of an outcome, event, or observation. 
5. **Naive Bayes**: It is an algorithm that learns the probability of every object, its features, and which groups they belong to. It works on conditional probability based on Bayes theorem. 

## Web Application
A Flask-based web app has been developed to allow users to input breast cancer feature data which is mean radius, mean texture, mean perimeter, mean area and mean concave points and get predictions from the trained machine learning models whether a tumor is benign or malignant. 

### Features:
- User-friendly form for entering input features.
- Real-time prediction output based on input data.
- Model comparison based on accuracy metrics.

### Live Demo:
http://127.0.0.1:5000/

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/breast-cancer-prediction.git
    cd breast-cancer-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the Breast Cancer Wisconsin dataset:
    You can download the dataset from [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and place it in the `dataset/` folder.

## Usage
1. Train the models by running the `train.py` script:
    ```bash
    python train.py
    ```

2. Start the Flask web application:
    ```bash
    python app.py
    ```

3. Open your web browser and visit `http://127.0.0.1:5000/` to interact with the app.

## Data
The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains the following attributes:

- **Radius**: Mean of distances from the center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter**: The perimeter of the tumor.
- **Area**: The area of the tumor.
- **Smoothness**: Local variation in radius lengths, and more.

The dataset contains 569 instances, with 30 features, and a target label indicating whether the tumor is benign or malignant.

## Results
The performance of each algorithm is evaluated based on accuracy. 

| Algorithm          | Accuracy on training data| Accuracy on Testing data| 
|--------------------|--------------------------| ------------------------|
| SVC                |         92.5%            |          90.3%          | 
| KNN                |         95.4%            |          91.2%          | 
| Random Forest      |         100%             |          94.7%          | 
| Logistic Regression|         94.9%            |          92.1%          |
| Naive Bayes        |         93.6%            |          93.8%          |

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

## Feedback
If you have any feedback, please reach out to me at [email](maryamkhansa0177@gmail.com)
