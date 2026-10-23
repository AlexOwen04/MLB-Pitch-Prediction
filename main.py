import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Load the statcast data
data = pd.read_csv("data/savant_data.csv", low_memory=False)

# #Display the first five rows
# print(data.head())

# #Display the Column names
# print(data.columns.tolist())

# #Check how many rows and columns
# print("Dataset size: ", data.shape)

# #Check for missing values
# print(data.isnull().sum())

# #Look at the different pitch outcomes
# print(data["description"].value_counts())

#Columns model will use to make predictions
features = [
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "stand",
    "p_throws"
]

#Keep only columns needed
data = data[features + ["description"]].copy()

#remove rows with missing info
data = data.dropna()

#Keep only common outcomes with enough examples
outcomes = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "hit_into_play"
]

data = data[data["description"].isin(outcomes)]

print(data["description"].value_counts)
print("Rows after cleaning:", len(data))

#Separate the inputs from the outcome we want to predict
X = data[features]
y = data["description"]

#identify text columns and numeric columns
categorical_features = [
    "pitch_type",
    "stand",
    "p_throws"
]

numeric_features = [
    "release_speed",
    "plate_x",
    "plate_z",
    "balls",
    "strikes"
]

#convert text categories into a format the model can use
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numbers",
            "passthrough",
            numeric_features
        )
    ]
)

#Build the complete machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Train the model
model.fit(X_train, y_train)

#Test it on data the model has not traind on
predictions = model.predict(X_test)

#evaluate the results
print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

labels = model.named_steps["classifier"].classes_

matrix = confusion_matrix(
    y_test,
    predictions,
    labels = labels
)

plt.figure(figsize=(9,6))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)

plt.xlabel("Predicted Outcome")
plt.ylabel("Acutal Outcome")
plt.title("MLB Pitch Outcome Prediction")
plt.tight_layout()
plt.show()