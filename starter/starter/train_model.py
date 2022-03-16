# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
import pickle

# Add code to load in the data.
data = pd.read_csv('data/clean_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

train.to_csv("data/train.csv", index= False)
test.to_csv("data/test.csv", index = False)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

with open('data/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

with open('data/lb.pickle', 'wb') as f:
    pickle.dump(lb, f)

# Train and save a model.
model = train_model(X_train, y_train)

model.save_model("model/model.json")
