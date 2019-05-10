import unittest
from axtrainer.xgboost import main as xgboost
from axtrainer.rfr import main as rfr
from axtrainer.weighted_model import main as weighted_model
import os
import glob

class TestStringMethods(unittest.TestCase):

    def test_xgboost(self):
        ax, best_parameters, results= xgboost()
        self.assertEqual(dict, type(results))
    def test_rfr(self):
        ax, best_parameters, results=  rfr()
        self.assertEqual(dict, type(results))
    def test_weighted_model(self):
        ax, best_parameters, results=  weighted_model()
        self.assertEqual(dict, type(results))
if __name__ == '__main__':
    unittest.main()