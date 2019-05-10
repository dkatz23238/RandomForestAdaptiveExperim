from axtrainer.xgboost import main
from pandas import DataFrame
import json
import datetime
import os

RES_PATH = os.environ.get("EXPERIMENT_RESULTS_PATH","experiment-results")

def save_results(results):
    ''' Save experiment to output'''
    if not os.path.exists(RES_PATH):
        os.mkdir(RES_PATH)
    isonow = datetime.datetime.now().isoformat()
    with open(os.path.join(RES_PATH, f"{isonow}.json"), "w" ) as f:
        f.write(json.dumps(results))
    return True

if __name__ == "__main__":
    
    ax, b , results = main()    
    if save_results(results):
        print("Complete!")

    print(f"Best Results {b}")