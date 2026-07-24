# Video Game Sales Prediction with Neural Networks

A machine learning project for predicting **global video game sales** using a fully connected **Artificial Neural Network (ANN / MLP)** trained on the `Video_Games.csv` dataset.

This project focuses on end-to-end tabular data preprocessing, missing value handling, categorical encoding, feature scaling, and deep learning-based regression for predicting `Global_Sales`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Neural Network Model](#neural-network-model)
- [Training Procedure](#training-procedure)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Results](#results)
- [Strengths of the Approach](#strengths-of-the-approach)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Requirements](#requirements)
- [License](#license)

---

## Project Overview

This repository contains a regression pipeline that predicts **video game global sales** from structured metadata such as:

- platform
- release year
- genre
- publisher
- regional sales
- critic scores
- user scores
- developer
- age rating

The core predictive model is a **Multi-Layer Perceptron (MLP)** built with **TensorFlow / Keras**.

Because the dataset contains both **numerical** and **categorical** variables as well as many **missing values**, a major part of the project is dedicated to robust preprocessing.

---

## Dataset

**File used:** `Video_Games.csv`

### Main columns
The dataset includes the following types of features:

- **Categorical Features**
  - `Name`
  - `Platform`
  - `Genre`
  - `Publisher`
  - `Developer`
  - `Rating`

- **Numerical Features**
  - `Year_of_Release`
  - `NA_Sales`
  - `EU_Sales`
  - `JP_Sales`
  - `Other_Sales`
  - `Critic_Score`
  - `Critic_Count`
  - `User_Score`
  - `User_Count`

- **Target Variable**
  - `Global_Sales`

### Dataset characteristics
- Mixed-type tabular dataset
- Contains missing values in both categorical and numerical columns
- Includes high-cardinality categorical features such as `Name` and `Publisher`
- Suitable for regression tasks

---

## Problem Statement

The goal of this project is to build a neural network that learns the relationship between video game metadata and **global sales performance**.

Formally:

Given a feature vector $X$, predict:

$$
y = \text{Global\_Sales}
$$

where:
- $X$ contains encoded and scaled game-related attributes
- $y$ is a continuous numerical target

This is a **supervised regression problem**.

---

## Project Workflow

The overall workflow of the project is:

1. Load the dataset
2. Identify missing values
3. Impute numerical and categorical features
4. Reduce high-cardinality categorical noise
5. Encode categorical variables
6. Handle outliers
7. Scale features
8. Split into train and test sets
9. Build and train the neural network
10. Evaluate model performance using regression metrics

---

## Data Preprocessing

Data preprocessing is one of the most important parts of this project.

### 1. Missing Value Handling

#### Numerical columns
Numerical missing values are imputed using **KNNImputer**, which estimates missing entries based on neighboring samples in feature space.

This was chosen instead of simple mean/median imputation to preserve more structure in the data.

#### Categorical columns
Categorical missing values are filled using **probability-based random sampling** from the existing distribution of each column.

This approach helps preserve the original category proportions better than filling with a single fixed label such as `"Unknown"`.

---

## Feature Engineering

### 1. High-Cardinality Category Reduction

Some categorical columns, especially:

- `Publisher`
- `Name`

contain many unique values.

To reduce sparsity and improve generalization:

- rare categories with frequency below a selected threshold were grouped into:
  - `Other`

This helps:
- reduce dimensionality
- avoid overly sparse one-hot vectors
- improve model stability

### 2. One-Hot Encoding

After category consolidation, categorical variables were transformed using **One-Hot Encoding**.

This converts text-based categories into machine-readable binary vectors.

### 3. Outlier Handling

For numerical columns, outliers were handled using **IQR-based clipping**:

- values below the lower bound were clipped
- values above the upper bound were clipped

This prevents extreme values from dominating the training process.

### 4. Feature Scaling

All numerical input features were standardized using **StandardScaler**.

This is especially important for neural networks because:
- it stabilizes training
- speeds up convergence
- improves gradient-based optimization

---

## Neural Network Model

The main predictive model is a **feedforward neural network** (MLP) implemented in **Keras**.

### General architecture
A typical architecture used in this project includes:

- Input layer
- Dense hidden layers with nonlinear activations
- Dropout layers for regularization
- Output layer with a single neuron for regression

### Example architecture
- Dense layer
- ReLU activation
- Dropout
- Dense layer
- ReLU activation
- Dropout
- Dense output layer with linear activation

### Why this model?
This model is suitable because:
- the dataset is tabular
- the target is continuous
- nonlinear relationships may exist among features
- dropout helps reduce overfitting

---

## Training Procedure

The model is trained on the processed dataset after train/test splitting.

### Steps
1. Separate features and target
2. Split data into training and testing sets
3. Fit the scaler on training data only
4. Transform both training and testing data
5. Train the neural network using training data
6. Monitor training and validation performance

### Regularization
To improve generalization, **Dropout** is used between hidden layers.

### Loss Function
Since this is a regression problem, a suitable loss such as:

- **Mean Squared Error (MSE)**

is used during training.

### Optimizer
Typical optimizers for this project include:

- `Adam`

because it usually performs well on tabular deep learning tasks.

---

## Evaluation Metrics

The model can be evaluated using standard regression metrics such as:

### Mean Squared Error
$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### Root Mean Squared Error
$$
RMSE = \sqrt{MSE}
$$

### Coefficient of Determination
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

These metrics provide complementary views of model quality:
- **MSE/RMSE** measure prediction error magnitude
- **R²** shows how much variance is explained by the model

---

## Project Structure

Example structure:
```bash
video-game-sales-ann/
│
├── data/
│   └── Video_Games.csv
│
├── notebooks/
│   └── video_games_neural_network.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── train.py
│
├── outputs/
│   ├── plots/
│   ├── trained_models/
│   └── metrics/
│
├── requirements.txt
└── README.md

