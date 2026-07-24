# Spotify Track Popularity Classification with ANN

A machine learning project for **predicting Spotify track popularity classes** using **feature engineering**, **tree-based feature importance**, and an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

This project transforms raw Spotify audio features into a structured classification pipeline that predicts whether a song belongs to one of three popularity levels:

- **Low**
- **Medium**
- **Hit**

It also includes a **simulation function** for predicting the popularity class probabilities of a **new hypothetical song**.

---

## Project Overview

The goal of this project is to build a complete machine learning workflow for Spotify music data, including:

- Data loading and initial exploration
- Missing value handling
- Duplicate removal
- Outlier filtering using Z-score
- Feature importance analysis with Random Forest
- Correlation analysis for multicollinearity auditing
- Target engineering for 3-class popularity prediction
- Feature engineering for musical/audio variables
- Data scaling with `StandardScaler`
- ANN model design, training, and evaluation
- New-song popularity simulation using the trained model

This project focuses on turning raw Spotify track information into a reliable **multi-class classification system**.

---

## Problem Definition

Instead of predicting raw popularity as a continuous value, this project converts the task into a **3-class classification problem**.

### Target Classes

A custom target variable called `target_class` is created using:

- the **track popularity**
- the **most important numerical driver feature**, selected automatically using a Random Forest regressor

### Class Logic

The three classes are generated using percentile- and median-based rules:

- **Class 2 (Hit):** high popularity and high driver-feature value
- **Class 1 (Medium):** moderate or mixed case
- **Class 0 (Low):** low popularity and low driver-feature value

This feature-driven labeling strategy makes the classification target more informative than using popularity alone.

---

## Dataset

The project expects a dataset file named:

```bash
1-spotify.csv
```

The code searches for the dataset in the following locations:

- `datasets/1-spotify.csv`
- `../datasets/1-spotify.csv`
- `/mnt/data/1-spotify.csv`
- `./1-spotify.csv`

### Example Expected Columns

Typical columns used in the project include:

- `track_id`
- `track_name`
- `artist_name`
- `genre`
- `key`
- `mode`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`
- `popularity`

---

## Main Workflow

The project is organized into seven main stages:

1. **Import Libraries**
2. **Load, Shuffle, and Initial EDA**
3. **Feature Importance and Driver Feature Selection**
4. **Multicollinearity Audit**
5. **Three-Class Target Creation and Full Preprocessing**
6. **ANN Design, Training, and Evaluation**
7. **Simulation for a New Song**

---

## 1) Import Libraries

The project uses the following major libraries:

- `NumPy`
- `Pandas`
- `Matplotlib`
- `SciPy`
- `scikit-learn`
- `TensorFlow / Keras`

These libraries support:

- numerical processing
- tabular data handling
- plotting
- outlier analysis
- preprocessing
- model building and evaluation

---

## 2) Load, Shuffle, and Initial EDA

The dataset is loaded into a Pandas DataFrame and then shuffled using a fixed random seed for reproducibility.

### Initial exploration includes:

- dataset shape
- first rows preview
- column names
- data types
- missing-value summary
- duplicate-row count
- descriptive statistics
- histogram of `popularity`

This stage helps identify data quality issues before modeling.

---

## 3) Feature Importance and Driver Feature Selection

To identify the strongest numerical feature associated with popularity, the project uses a:

- **RandomForestRegressor**

### Steps

- Only numerical features are selected
- Metadata columns are excluded
- Missing numerical values are filled with medians
- A sample of up to 10,000 rows is used for efficiency
- Feature importances are computed and ranked

### Output

The most important numerical feature is stored as:

```python
driver_feature
```

This selected feature is later used in target engineering.

---

## 4) Multicollinearity Audit

To examine relationships among numerical variables, a correlation matrix is built.

### This stage includes:

- numerical correlation matrix display
- correlation heatmap with Matplotlib
- identification of highly correlated feature pairs where:

```python
|correlation| > 0.85
```

This helps reveal redundant variables and potential multicollinearity problems.

---

## 5) Three-Class Target Creation and Full Preprocessing

This is the core preprocessing stage of the project.

### A) Duplicate Removal

Exact duplicate rows are removed to improve data quality.

### B) Missing Value Handling

#### Numerical features
Missing values are filled using the **median**.

#### Categorical features
Missing values are filled using the **mode**.

This strategy avoids unnecessary row deletion and preserves the dataset size.

### C) Target Engineering

A custom three-class target called `target_class` is created using:

- the 70th percentile of `popularity`
- the 70th percentile of the selected `driver_feature`
- the medians of both variables

This generates classes representing:

- low-popularity tracks
- medium-popularity tracks
- hit songs

### D) Outlier Removal

Outliers are filtered using **Z-score analysis**.

Rows are kept only if all selected numerical features satisfy:

```python
|Z| < 3
```

This reduces the influence of extreme values on the model.

### E) Feature Engineering

Several transformations are applied to improve representational quality.

#### 1. Cyclical encoding for `key`
Because musical key is cyclic rather than linear, it is encoded using sine and cosine:

```python
key_sin = sin(2π * key / 12)
key_cos = cos(2π * key / 12)
```

The original `key` column is then removed.

#### 2. Frequency encoding for `genre`
The categorical `genre` column is converted into a numerical representation using normalized frequency counts.

A mapping dictionary is stored for later use in new-song simulation:

```python
genre_frequency_map
```

The original `genre` column is then dropped.

#### 3. Energy-Loudness Ratio
A new engineered feature is created:

```python
energy_loudness_ratio = energy / (abs(loudness) + 1)
```

This combines two audio characteristics into a potentially more informative signal.

### F) Feature Matrix and Labels

The following columns are removed before training:

- `track_id`
- `track_name`
- `artist_name`
- `index`
- `popularity`
- `target_class`

The final feature matrix contains only numerical variables.

### G) Train / Validation / Test Split

The data is split into:

- **training set**
- **validation set**
- **test set**

The split is stratified to preserve class distribution.

### H) Feature Scaling

A `StandardScaler` is fitted only on the training set and then applied to:

- training data
- validation data
- test data

This ensures no data leakage.

---

## 6) ANN Design, Training, and Evaluation

The classification model is an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

### Network Architecture

The ANN includes:

- Dense input layer with **512 neurons**
- `LeakyReLU`
- `BatchNormalization`
- `Dropout(0.30)`

followed by:

- Dense layer with **256 neurons**
- `tanh`
- `BatchNormalization`
- `Dropout(0.25)`

followed by:

- Dense layer with **64 neurons**
- `LeakyReLU`
- `Dropout(0.20)`

and finally:

- Dense output layer with **3 neurons**
- `softmax` activation

### Compilation Settings

The model is compiled with:

- **Optimizer:** `Adam`
- **Learning rate:** `0.001`
- **Loss:** `sparse_categorical_crossentropy`
- **Metric:** `accuracy`

### Class Imbalance Handling

To address class imbalance, the project computes **balanced class weights** using:

```python
compute_class_weight()
```

These weights are passed into Keras during training.

### Training Strategy

Two training scenarios are used:

#### 1. Subset experiment
A small subset (5% of training data) is used for a quick baseline experiment.

#### 2. Full-scale training
The full training set is used for the final model.

### Callbacks

Training stability is improved with:

#### EarlyStopping
Stops training when validation loss stops improving.

#### ReduceLROnPlateau
Reduces the learning rate when validation accuracy plateaus.

These callbacks help prevent overfitting and improve convergence.

### Evaluation Metrics

The final ANN is evaluated on the test set using:

- **Accuracy**
- **Macro F1-score**
- **Classification report**
- **Normalized confusion matrix**

Predicted probabilities are also generated for further analysis.

---

## 7) New Song Simulation

One of the most useful parts of this project is the function:

```python
simulate_new_song(raw_song)
```

This function predicts the class probabilities of a **new hypothetical Spotify track**.

### What the simulation function does

It:

- converts a raw Python dictionary into a one-row DataFrame
- fills missing numerical values
- applies cyclical encoding for `key`
- applies stored frequency encoding for `genre`
- creates `energy_loudness_ratio`
- removes metadata columns
- reorders columns to match training features
- scales the input using the training scaler
- predicts class probabilities using the trained ANN

### Output

The function returns a Pandas Series with probabilities for:

- `Low`
- `Medium`
- `Hit`

### Example

```python
example_song = {
    "genre": "pop",
    "key": 5,
    "mode": 1,
    "danceability": 0.72,
    "energy": 0.81,
    "loudness": -5.8,
    "speechiness": 0.06,
    "acousticness": 0.18,
    "instrumentalness": 0.01,
    "liveness": 0.12,
    "valence": 0.65,
    "tempo": 124,
    "duration_ms": 210000
}

simulate_new_song(example_song)
```

### Example Output

```python
Low       0.12
Medium    0.35
Hit       0.53
dtype: float32
```

You can also extract the final predicted class using:

```python
result = simulate_new_song(example_song)
predicted_class = result.idxmax()
print("Predicted class:", predicted_class)
```

---

## Project Highlights

### Strengths of the project

- Full end-to-end machine learning pipeline
- Custom target engineering
- Feature-driven class labeling
- Robust preprocessing strategy
- Meaningful musical feature engineering
- ANN-based multi-class classification
- Class imbalance handling
- New-song simulation support
- Reproducible workflow with fixed random seeds

---

## Requirements

Install the required packages before running the notebook or script:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Place the dataset
Make sure `1-spotify.csv` is available in one of the expected directories.

### 2. Run the notebook or script
Execute the cells in order:

1. imports  
2. data loading  
3. feature importance  
4. correlation audit  
5. preprocessing  
6. ANN training  
7. simulation  

### 3. Predict a new song
Use the `simulate_new_song()` function with your own song features.

---

## Example Use Case

This project can be used for:

- estimating whether a song is likely to become a **hit**
- testing hypothetical combinations of audio features
- understanding which Spotify-style features matter most
- experimenting with ANN-based music popularity modeling

---

## Possible Future Improvements

There are several ways this project can be extended:

- compare ANN with XGBoost, LightGBM, or CatBoost
- tune ANN hyperparameters more systematically
- use embeddings for high-cardinality categorical features
- apply probability calibration
- use cross-validation for more robust evaluation
- create a web app for interactive song simulation
- add SHAP or permutation importance for interpretability
- test alternative target engineering strategies

---

## Reproducibility Notes

This project sets random seeds for:

- `NumPy`
- `TensorFlow`

This helps improve reproducibility, though exact neural network results may still vary slightly depending on hardware and backend behavior.

---

## Repository Structure

```bash
spotify-popularity-classification/
│
├── README.md
├── notebook.ipynb
├── 1-spotify.csv
└── requirements.txt
```

If your dataset is stored elsewhere, update the search paths in the code.

---

## Conclusion

This project demonstrates a complete machine learning workflow for **Spotify track popularity classification**, combining:

- data cleaning
- feature selection
- feature engineering
- neural network modeling
- model evaluation
- real-world simulation for unseen songs

It is a practical example of how structured audio features can be transformed into a predictive system for music popularity categories.
