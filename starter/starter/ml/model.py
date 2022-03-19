import xgboost as xgb


# Optional: implement hyperparameter tuning.
def prepare(X, y):
    return xgb.DMatrix(X, label=y)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    dtrain = prepare(X_train, y_train)
    model = xgb.train(params=param, dtrain=dtrain)
    return model


def load_xgb_model(path):
    model_xgb = xgb.Booster()
    model_xgb.load_model(path)
    return model_xgb
