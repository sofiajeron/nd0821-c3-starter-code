# Model Card

Author: Sofia Jeronimo
Last updated: Mar 2022

## Model Details

The data set is a collection of census data, with ~32k records and 15 columns, being the "salary" the label we want to predict.
It is a binary classification problem which the label "salary" has the values:
- <=50K
- =>50k
whether a given person earns more or less than 50k a year. 
- 
The model used is a XGBoost model and was trained on 80% of the dataset, the remaining 20% was used to test the data.
The serialized model is stored in ```model/model.json```.
The model is trained on 70% of dataset, 15% is reserved for validation and 15% for tests.


## Intended Use
This work was done as part of the project assessment for Udacity nano degree program. 
It includes a CI/CD pipeline for Machine Learning using Github actions for CI process, FastAPI to develop the whole inference API and Heroku to deploy the application in the web and includes the CD process.
We used the DVC to track the data changes, store the models and respective performance metrics.
DVC was integrated in S3, and used to remotely store data and model artifacts.

## Training Data
The training data includes ~26k records and stored ```data/train.csv```. The categorical data were transformed to one hot encoding and we processed the labels with label binarizer (lb) encoder, both encoder were stored in ```data/encoder.pickle``` and ```data/lb.pickle```.

## Test Data
The evaluation data includes ~6k records and stored ```data/test.csv```.

## Metrics
The model results can be showed with the code ```dvc metrics show ```.
The average precision, recall and fbeta were calculated and stored in ```model/``` and were the following:

```json
{"precision": 0.7624172185430463,
  "recall": 0.5941935483870968, 
  "fbeta": 0.6678752719361857}
```

The performance discriminated per category value was calculated in order to check for model bias and fairness and can be found in ```model/slice_output.txt```.

## Ethical Considerations

The Census Bureau data is publicly available in the UCI repository and with individuals' consent. 
The data is not enough to have solid conclusions about the fairness predictions across ethnicities, more data is needed for that analysis.


## Caveats and Recommendations

For this project was used a very simple model without extensive hyperparameter tuning, and mainly focus on the automatization of the processes with CI/CD process, so continuous work should be done to archieve a better model that have an improvement of performance as well as predicts fairly across the dataset. 