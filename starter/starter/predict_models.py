from ml.evaluation import metrics_slice
from ml.evaluation import inference, compute_model_metrics, save_model_metrics
from ml.model import load_xgb_model
from ml.data import process_data
import pickle
import pandas as pd

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

with open('data/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

with open('data/lb.pickle', 'rb') as f:
    lb = pickle.load(f)

test = pd.read_csv("data/test.csv")

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

xgb_model = load_xgb_model("model/model.json")
preds = inference(xgb_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
save_model_metrics("precision", precision)
save_model_metrics("recall", recall)
save_model_metrics("fbeta", fbeta)
metrics_slice(cat_features, preds, y_test, "model/", test)