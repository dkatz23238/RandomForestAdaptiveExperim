# ax-container

Dependens:

Python 3.7+

This project does the following:
- Loads a machine learning data set
- Instantiates an Adaptive Experimentation service loop
- Trains and tunes an xgboost.XGBRegressor model on a dataset. The dataset provided by default is house pricing data in buenos aires.

Note: the data set must be ready to process by an sklearn or xgboost algorithm.

## Quickstart

Review the enviornment varianles and if needed update the dataset.csv with custom data set. Remember to update the TARGET enviornment variable with the name of the target variable column.

The following enviornment variables can be used and modifed in docker-compose.yml:
 - N_TRIALS: the amount of trials to run by adaptive experimentation
 - DATASET_PATH: the path to the machine learning dataset
 - DATASET_TARGET_NAME: the name of the column that contains the target variable

``` docker-compose up```

or without docker:

cd to app
``` python -m pip install -r requirements.txt && python run.py```

The hyper-parameter tunining process will be printed to screen or loged to the docker compose logs.