import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from haversine import haversine
pd.options.display.max_columns = 999


BUENOS_AIRES_CENTER_COORDS = (-34.6037,-58.3816)
ABASTO_LOCATION = (-34.36779, -58.242339)
VTE_LOPEZ_STATION = (-34.5248, -58.4728)
VILLA_1_11_14 = (-34.650291, -58.440893)

def remove_outliers(train_set):
    MU = train_set.DOLARES_M2.mean() 
    SIGMA = train_set.DOLARES_M2.std()
    MAX = MU + (2*SIGMA)
    MIN = MU - (2 * SIGMA)
    return train_set[(MIN < train_set.DOLARES_M2) & (train_set.DOLARES_M2 < MAX)]

# Add custom features
def get_distance_list(train_set, coords=BUENOS_AIRES_CENTER_COORDS):
    d_l = []
    for r in train_set.iterrows():
        LON = r[1].LON
        LAT = r[1].LAT
        d1 = (LAT, LON)
        center = coords

        distance_to_city_center = haversine(d1, center)
        d_l.append(distance_to_city_center)
    return d_l

def clean_ambientes(x):
    if x > 4:
        return ">4"
    else:
        return str(x)


df = pd.read_csv("./cleaned_depto_ventas.csv")


df["AMBIENTES"] = df.AMBIENTES.apply(clean_ambientes)

for col in df:
    if df[col].dtype == object:
        df[col] = df[col].str.strip().str.upper()

train_cols = [
             'M2',
             'DOLARES',
             'AMBIENTES',
             'ANTIGUEDAD',
             'ORIENT',
             'BAULERA',
             'COCHERA',
             'BAÑOS',
             'LAVADERO',
             'TERRAZA',
             'COMUNA',
             'LON',
             'LAT',
             'AÑO'
            ]

train_set = df[train_cols]

# Make more features
train_set["DISTANCE_TO_CENTER_KM"] = np.log1p(get_distance_list(train_set, coords=BUENOS_AIRES_CENTER_COORDS))
train_set["DISTANCE_TO_ABASTO_KM"] = np.log1p(get_distance_list(train_set, coords=ABASTO_LOCATION))
train_set["DISTANCE_TO_VTELPZ_KM"] = np.log1p(get_distance_list(train_set, coords=VTE_LOPEZ_STATION))
train_set["DISTANCE_TO_VILLA_11114_KM"] = np.log1p(get_distance_list(train_set, coords=VILLA_1_11_14))
train_set["DOLARES_M2"] = (train_set.DOLARES / train_set.M2).replace(np.inf, np.nan)

NUMERICAL_COLS = ['ANTIGUEDAD','DISTANCE_TO_CENTER_KM','DISTANCE_TO_ABASTO_KM', 'DISTANCE_TO_VTELPZ_KM','DISTANCE_TO_VILLA_11114_KM']
# NUMERICAL_COLS = ['ANTIGUEDAD','DISTANCE_TO_CENTER_KM', 'DISTANCE_TO_VILLA_11114_KM']


CAT_COLS = ["BAULERA","COMUNA", 'AÑO','AMBIENTES']
TARGET = "DOLARES_M2"

train_set = remove_outliers(train_set)

processed_set = pd.concat(
    (train_set[NUMERICAL_COLS],
    pd.get_dummies(train_set[CAT_COLS])
    ),
    axis=1)

# Definte Target Variable
processed_set["target"] = train_set[TARGET].apply(np.log1p)

# drop na
processed_set = processed_set.dropna()
print("DF SHAPE: %s" % str(processed_set.shape))
X = processed_set[[i for i in processed_set.columns if i != "target"]]
Y = processed_set["target"]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.15)

DATA_SET = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)