from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from axtrainer.data import DATA_SET_DICT
# from axtrainer.logger import *
from ax import ChoiceParameter, ParameterType
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
# from axtrainer.trainer import DATA_SET_DICT
import os


def make_parameter(name, ptype, bounds, value_type):
    ''' Creates a parameter dictionary to be used in ax.create_experiment'''
    if ptype == "range":
        return dict(name=name, type=ptype, bounds=bounds, value_type=value_type)
    elif ptype == "choice":
        return dict(name=name, type=ptype, values=bounds, value_type=value_type)


def train_and_return_score(Model= xgb.XGBRegressor, **kwargs):
    ''' Convinience function to train model and return score'''
    X_train, X_test, y_train, y_test = DATA_SET_DICT["X_train"], DATA_SET_DICT[
        "X_test"], DATA_SET_DICT["y_train"], DATA_SET_DICT["y_test"]
    # Instantiate model with keyword arguments
    rf = RandomForestRegressor(n_estimators=20)
    model = Model(n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    pred_xg = model.predict(X_test)
    # pred_rf = rf.predict(X_test)
    # pred = np.mean([pred_rf, pred_xg], axis=0)
    _score = mean_squared_error(y_test, pred_xg)
    # print("MODEL SCORE: %s " % _score)
    return _score

PARAMETERS = [

    make_parameter("n_estimators", "range", [0, 300], "int"),
    make_parameter("alpha", "range", [0,100], "int"),
    make_parameter("max_depth", "range", [0,10], "int"),
    make_parameter("learning_rate", "range", [0.01, 0.1], "float"),
    make_parameter("reg_alpha", "range", [0., 0.9], "float"),
    make_parameter("subsample", "range", [0.5, 0.99], "float"),
    make_parameter("gamma", "range", [0, 0.99], "float"),
    make_parameter("reg_lambda", "range", [0, 10], "float")     
]

def main(parameters=PARAMETERS):
    ''' Main experiment loop'''

    ax = AxClient()


    ax.create_experiment(
        name="RFR",
        parameters=parameters,
        objective_name="mean_squared_error",
        minimize=True,
    )
    N_TRIALS = os.environ.get("N_TRIALS", 15)

    for _ in range(N_TRIALS):
        parameters, trial_index = ax.get_next_trial()
        print(f"Trial Index: {trial_index}")
        print(f"Parameters: {parameters}")

        ax.complete_trial(
            trial_index=trial_index,
            raw_data=train_and_return_score(
                n_estimators=parameters["n_estimators"],
                alpha=parameters["alpha"],
                max_depth=parameters["max_depth"],
                learning_rate=parameters["learning_rate"],
                reg_alpha=parameters["reg_alpha"],
                reg_lambda=parameters["reg_lambda"],
                subsample=parameters["subsample"],
                gamma=parameters["gamma"],
                )
            )

    best_parameters, metrics = ax.get_best_parameters()
    return ax, best_parameters, metrics
