from flask import Flask, render_template
from flask import Markup
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import pandas as pd

import json
import glob

app = Flask(__name__)
PATH_TO_EXPERIMENT_JSON = glob.glob('/home/david/Desktop/ax-container/app/experiment-results/*.json')[0]
with open(PATH_TO_EXPERIMENT_JSON, encoding="utf-8") as f:
  jsondata = json.loads(f.read())


PARAM1 = "w1"
PARAM2 = "w2"

@app.route('/3d')
def hello(name=None):
    ''' Interactive 2d graph'''
    vis_data = [{"id":i,"z":np.log1p(j*100),"style":j} for i,j in enumerate(jsondata["score"])]
    for i, el in enumerate(vis_data):
        el["x"] = np.log1p(jsondata["parameters"][i][PARAM1]*100)
        el["y"] = np.log1p(jsondata["parameters"][i][PARAM2]*100)
    return render_template('3d.html', jsondata=Markup(vis_data), param1=PARAM1, param2=PARAM2)


@app.route('/2d')
def twod(name=None):
    ''' Interactive 3d graph'''
    vis_data = [ { "id": i, "y": np.log1p(j*100)} for i, j in enumerate(jsondata["score"]) ]
    for i, el in enumerate(vis_data):
        el["x"] = np.log1p(jsondata["parameters"][i][PARAM1]*100)
    return render_template('2d.html', jsondata=Markup(vis_data))

@app.route('/3ddd')
def hellod(name=None):
    ''' Interactive #d graph'''
    vis_data = [{"z":j, "style":j} for i,j in enumerate(jsondata["score"])]
    x = []
    y = []
    z = []
    for i, el in enumerate(vis_data):
        el["x"] = np.log1p(jsondata["parameters"][i][PARAM1]*100)
        el["y"] = np.log1p(jsondata["parameters"][i][PARAM2]*100)
        x.append(el["x"])
        y.append(el["y"])
        z.append(el["z"])

    train = pd.DataFrame({"x":x,"y":y})
    print(train)

    clf = LinearRegression()
    clf.fit(train, z)
    pred_set = pd.DataFrame({"x":np.random.randint(0,150,size=400),"y":np.random.randint(0,100,size=400) })
    print(pred_set)
    Z = [i for i in clf.predict(pred_set)]
    
    return_dict = [{"style": Z[i], "z":Z[i], "x":pred_set.x.tolist()[i], "y":pred_set.y.tolist()[i],} for i in range(400)]
    return render_template('3d.html', jsondata=Markup(return_dict), param1=PARAM1, param2=PARAM2)

    
if __name__=='__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)