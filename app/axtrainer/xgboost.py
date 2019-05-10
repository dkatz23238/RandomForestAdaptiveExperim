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
        return dict(name=name, type=ptype, values=bounds)

# Function to return our target cost function and optimize parameters with ax.Client
def train_and_return_score(**kwargs):
    ''' Convinience function to train model and return score'''
    if PROBLEM_TYPE == "REGRESSION":
        Model = xgb.XGBRegressor
    elif PROBLEM_TYPE == "CLASSIFICATION":
        Model = xgb.XGBClassifier
    
   
    X_train, X_test, y_train, y_test = DATA_SET_DICT["X_train"], DATA_SET_DICT[
        "X_test"], DATA_SET_DICT["y_train"], DATA_SET_DICT["y_test"]
    # Instantiate model with keyword arguments
    estimators = [
        Model(n_jobs=-1,gpu_id=0, **kwargs)
    ]
    for model in estimators:
        model.fit(X_train, y_train)
    
    preds = list(model.predict(X_test) for model in estimators)
    
    # Weighted sum of models
    _score=0.0
    for pred in preds:
        try:
            _score += mean_squared_error(y_test, pred)
        except:
            _score += 0.0
    
    # print("MODEL SCORE: %s " % _score)
    return _score / float(len(preds))

PARAMETERS = [
    make_parameter("n_estimators", "range", [10, 300], "int"),
    make_parameter("alpha", "range", [0,100], "int"),
    make_parameter("max_depth", "range", [0,10], "int"),
    make_parameter("learning_rate", "range", [0.01, 0.1], "float"),
    {"name":"booster", "type": "fixed", "value":"gtree"}   ,
    # make_parameter("reg_alpha", "range", [0., 0.9], "float"),
    # make_parameter("subsample", "range", [0.5, 0.99], "float"),
    # make_parameter("gamma", "range", [0, 0.99], "float"),
    make_parameter("reg_lambda", "range", [0, 10], "float") ,
    {"name":"gpu_id", "type":"fixed", "value":0},
    {"name":"max_bin", "type":"fixed", "value":16},
    {"name":"tree_method", "type":"fixed", "value":'gpu_hist'}
]


CONSTRAINTS = []
def main(parameters=PARAMETERS, parameter_constraints=CONSTRAINTS):
    ''' Main experiment loop'''

    results = {"trial_index":[], "parameters":[], "score":[]}

    ax = AxClient()


    ax.create_experiment(
        name="RFR",
        parameters=parameters,
        parameter_constraints=CONSTRAINTS,
        objective_name="mse",
        minimize=True,
    )
    N_TRIALS = int(os.environ.get("N_TRIALS", 10))

    for _ in range(N_TRIALS):
        parameters, trial_index = ax.get_next_trial()
        # print(f"Trial Index: {trial_index}")
        # print(f"Parameters: {parameters}")
        calculation = train_and_return_score(
                n_estimators=parameters["n_estimators"],
                alpha=parameters["alpha"],
                max_depth=parameters["max_depth"],
                learning_rate=parameters["learning_rate"],
                # reg_alpha=parameters["reg_alpha"],
                reg_lambda=parameters["reg_lambda"],
                # subsample=parameters["subsample"],
                # gamma=parameters["gamma"],
        )

        ax.complete_trial(
            trial_index=trial_index,
            raw_data= calculation
            )
        # print(f"Calculation {np.sqrt(calculation)}")
        results["trial_index"].append(trial_index)
        results["parameters"].append(parameters)
        results["score"].append(calculation)
    

    best_parameters, _ = ax.get_best_parameters()
    return ax, best_parameters, results
