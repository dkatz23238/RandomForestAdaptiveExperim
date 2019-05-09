from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from axtrainer.data import DATA_SET_DICT
from sklearn.linear_model import LinearRegression
# from axtrainer.logger import *
from ax import ChoiceParameter, ParameterType
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score
# from axtrainer.trainer import DATA_SET_DICT
import os

PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE", "REGRESSION")

# Helper function for parameter handling
def make_parameter(name, ptype, bounds, value_type):
    ''' Creates a parameter dictionary to be used in ax.create_experiment'''
    if ptype == "range":
        return dict(name=name, type=ptype, bounds=bounds, value_type=value_type)
    elif ptype == "choice":
        return dict(name=name, type=ptype, values=bounds, value_type=value_type)

# Function to return our target cost function and optimize parameters with ax.Client
def train_and_return_score(w1=1/3.0,w2=1/3.0,w3=1/3.0, **kwargs):
    ''' Convinience function to train model and return score'''
    if PROBLEM_TYPE == "REGRESSION":
        Model = xgb.XGBRegressor
    elif PROBLEM_TYPE == "CLASSIFICATION":
        Model = xgb.XGBClassifier
    
   
    X_train, X_test, y_train, y_test = DATA_SET_DICT["X_train"], DATA_SET_DICT[
        "X_test"], DATA_SET_DICT["y_train"], DATA_SET_DICT["y_test"]
    # Instantiate model with keyword arguments
    estimators = [
        RandomForestRegressor(n_estimators=30),
        Model(n_jobs=-1,gpu_id=0, **kwargs)
    ]
    for model in estimators:
        model.fit(X_train, y_train)
    
    preds = np.array(list(model.predict(X_test) for model in estimators))
    
    # Weighted sum of models
    preds = np.array((w1,w2)) @ preds

    _score = mean_squared_error(y_test, preds)
    # print("MODEL SCORE: %s " % _score)
    return _score

PARAMETERS = [
    make_parameter("w1", "range", [0, .99], "float"),
    make_parameter("w2", "range", [0, .99], "float"),

]

CONSTRAINTS = ["w1 + w2 <= 1.0",]

def main(parameters=PARAMETERS, parameter_constraints=CONSTRAINTS):
    ''' Main experiment loop'''

    ax = AxClient()


    ax.create_experiment(
        name="RFR",
        parameters=parameters,
        objective_name="mean_square_err",
        parameter_constraints=CONSTRAINTS,
        minimize=True,
    )
    N_TRIALS = int(os.environ.get("N_TRIALS", 50))

    for _ in range(N_TRIALS):
        parameters, trial_index = ax.get_next_trial()
        print(f"Trial Index: {trial_index}")
        print(f"Parameters: {parameters}")

        ax.complete_trial(
            trial_index=trial_index,
            raw_data=train_and_return_score(
                w1=parameters["w1"],
                w2=parameters["w2"],
                
                )
            )

    best_parameters, metrics = ax.get_best_parameters()
    return ax, best_parameters, metrics
