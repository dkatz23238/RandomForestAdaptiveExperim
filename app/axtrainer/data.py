import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = os.environ.get("DATASET_PATH", "./dataset.csv")
DATASET_TARGET_NAME = os.environ.get("DATASET_TARGET_NAME", "target")

processed_set = pd.read_csv(DATASET_PATH)
processed_set = processed_set.dropna()


X = processed_set[[i for i in processed_set.columns if i != DATASET_TARGET_NAME]]
Y = processed_set[DATASET_TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)

DATA_SET_DICT = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)