import pytest
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import react_app
import json



class MyTestCase(unittest.TestCase):

    def setUp(self):
        react_app.app.testing = True
        self.app = react_app.app.test_client()

    def test_home(self):
        response = self.app.get('/home')

    def test_training(self):
        response = self.app.get('/training')
    
