# MLB Pitch Outcome Prediction

## Overview

This project uses Python and machine learning to predict the outcome of an MLB pitch based on historical Statcast data. The goal is to analyze pitch characteristics, identify patterns, and evaluate how accurately a machine learning model can classify pitch outcomes.

This project demonstrates data cleaning, exploratory data analysis, machine learning, model evaluation, and data visualization.

## Objectives

- Analyze MLB pitch-by-pitch data from Baseball Savant.
- Clean and prepare data for machine learning.
- Explore relationships between pitch velocity, pitch location, pitch type, and pitch outcomes.
- Train a machine learning model to classify pitch outcomes.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Visualize prediction results using a confusion matrix.

## Technologies Used

- **Python** — Main programming language
- **pandas** — Data loading and preparation
- **scikit-learn** — Machine learning and model evaluation
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical charts and confusion matrix visualization

## Dataset

The project uses pitch-level MLB Statcast data from [Baseball Savant](https://baseballsavant.mlb.com/en/statcast_search).

The dataset includes information such as:

- Pitch type
- Pitch velocity
- Horizontal and vertical pitch location
- Ball and strike count
- Batter handedness
- Pitcher handedness
- Pitch outcome

The target variable is the recorded pitch outcome.

## Machine Learning Approach

A Random Forest classifier is used to predict one of five pitch outcomes:

- Ball
- Called strike
- Swinging strike
- Foul
- Hit into play

The model uses pitch characteristics as input features. Categorical variables are encoded using `OneHotEncoder`, while numeric variables are passed through the preprocessing pipeline.

The dataset is divided into training and testing sets to evaluate performance on data not used during training.

## Model Evaluation

The model is evaluated using:

- **Accuracy:** Percentage of predictions that are correct.
- **Precision:** How often predictions for an outcome are correct.
- **Recall:** How many actual examples of an outcome the model identifies.
- **F1-score:** A balance between precision and recall.
- **Confusion matrix:** A comparison of actual and predicted outcomes.

Actual performance results will be added after training and evaluating the model.

## Project Structure

```text
MLB_Pitch_Prediction/
├── data/
│   └── statcast.csv
├── main.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone or download this repository.
2. Install Python.
3. Open a terminal in the project directory.
4. Install the required packages:

```bash
py -m pip install -r requirements.txt
```

5. Download Statcast pitch-level data from Baseball Savant.
6. Place the CSV file in the `data` folder and name it `statcast.csv`.
7. Run the program:

```bash
py main.py
```

The program will prepare the data, train the machine learning model, display evaluation metrics, and generate a confusion matrix.

## Future Improvements

- Compare Random Forest with Logistic Regression and Decision Tree models.
- Investigate which pitch characteristics are most useful for prediction.
- Improve evaluation using chronological or game-based train/test splits.
- Build an interactive dashboard using Streamlit.
- Test model performance on pitches from a later time period.

## Data Source

[Baseball Savant — Statcast Search](https://baseballsavant.mlb.com/en/statcast_search)

## Author

Developed as a personal data science and machine learning project to explore MLB analytics and predictive modeling.
