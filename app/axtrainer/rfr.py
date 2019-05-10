from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from axtrainer.data import DATA_SET_DICT
from ax import ChoiceParameter, ParameterType
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

def train_and_return_score(Model=RandomForestRegressor, **kwargs):
    ''' Convinience function to train model and return score'''
    X_train, X_test, y_train, y_test = DATA_SET_DICT["X_train"], DATA_SET_DICT[
        "X_test"], DATA_SET_DICT["y_train"], DATA_SET_DICT["y_test"]
    # Instantiate model with keyword arguments
    model = Model(n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    _score = np.sqrt(mean_squared_error(y_test, pred))
    # print("MODEL SCORE: %s " % _score)
    return _score

PARAMETERS = [

    make_parameter("n_estimators", "range", [0, 300], "int"),
    make_parameter("bootstrap", "choice", [True, False], "bool"),
    make_parameter("max_depth", "range", [2, 20], "int")
    ]

def main(parameters=PARAMETERS):
    ''' Main experiment loop'''
    results = {"trial_index":[], "parameters":[], "score":[]}
    ax = AxClient()
    ax.create_experiment(
        name="RFR",
        parameters=parameters,
        objective_name="mean_squared_error",
        minimize=True,
    )
    N_TRIALS = int(os.environ.get("N_TRIALS", 10))

    for _ in range(N_TRIALS):
        parameters, trial_index = ax.get_next_trial()
        # print(f"Trial Index: {trial_index}")
        # print(f"Parameters: {parameters}")
        calculation = train_and_return_score(
                min_samples_split=parameters["max_depth"],
                n_estimators=parameters["n_estimators"],
                bootstrap=parameters["bootstrap"])
        # print(f"Calculation: {calculation}")
        ax.complete_trial(
            trial_index=trial_index,
            raw_data=calculation)
        results["trial_index"].append(trial_index)
        results["parameters"].append(parameters)
        results["score"].append(calculation)
    
    best_parameters, _ = ax.get_best_parameters()
    return ax, best_parameters, results
